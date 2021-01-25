#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from braceexpand import braceexpand
from scipy.ndimage.measurements import label 
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import os, pdb, sys, pickle
from argparse import ArgumentParser
from termcolor import colored, cprint
from io import StringIO
from pathlib import Path
import pandas as pd

from utils import *
from data import Geometry 
from neuralDose.net.Unet3D import UNet3D 
from neuralDose.data.datamodule import Transform 
from neuralDose.utils import MyModelCheckpoint


class PencilBeam():
    def __init__(self, hparam, data, roi_skinName='ITV_skin'):
        self.hparam = hparam
        self.roi_skinName = roi_skinName 
        cprint(f'using skin name {roi_skinName} to parsing PointsPosition.txt; pls confirm the skin name is right', 'red')
        self.data = data
        self.geometry = Geometry(data)
        self.dict_randomApertures = None

        # set dose grid coords
        self._set_points_positions()

        # set corner of maximum skine rect
        self._set_skin_coords()

        # {beam_id: deposition matrix (#doseGrid, #beamBixels)}
        self.dict_deps = convert_depoMatrix_to_tensor(self.data.dict_beamID_Deps, hparam.device)
        
    def _load_randApert(self):
        if self.dict_randomApertures == None:
            # load saved random apertures
            save_path = Path(self.hparam.patient_ID).joinpath('dataset/dict_randomApertures.pickle')
            if os.path.isfile(save_path):
                self.dict_randomApertures = unpickle_object(save_path)
            else:
                raise ValueError

    def _set_skin_coords(self): 
        '''get corner of maximum skin rectangle  '''
        dose_grid = self.dose_grid[:, 0:2]
        dose_grid = dose_grid.round().astype(np.uint)

        self.skin_lefttop = OrderedBunch({'x':dose_grid[:,0].min(), 'y':dose_grid[:,1].min()})
        self.skin_rightbot= OrderedBunch({'x':dose_grid[:,0].max(), 'y':dose_grid[:,1].max()})

    def _set_points_positions(self):
        '''get dose_grid coords, from pointsPosition.txt
        return: coords (#points, 3=[x,y,z])
        '''
        with open(self.hparam.pointsPosition_file, "r") as f:
            lines = f.readlines()
        
        # get begin and end lines in pointsPosition.txt for roi_skin
        begin, end = np.inf, np.inf
        for i, line in enumerate(lines):
            if self.roi_skinName in line or self.roi_skinName.replace('skin', 'SKIN') in line: 
                begin = i+1
            if i>=begin and ':' in line: # another organ 
                end = i
            if i>=begin and i+1==len(lines):
                end = i+1
        
        # parse each line for x,y,z coords
        coords = []
        lines = lines[begin:end]
        for line in lines:
            coord = [float(x) for x in line.split()]  # split uses dafault delimiter: space and \n 
            coords.append(coord)
        coords = np.asarray(coords)
        assert len(coords) == self.data.get_pointNum_from_organName(self.roi_skinName)

        # sort pointPositions to ensure the consistent ordering with deposition.txt: z decending, y decending, x ascending  
        sort_index = np.lexsort((coords[:,0], -coords[:,1], -coords[:,2]))  # NOTE TODO: the PTVs have duplicate begin and end slices
        coords = coords[sort_index]
        
        # change zs from physic coords (e.g. 3.6) to image coords (e.g. 0)
        zs = np.array([float(z)/10 for z in self.data.Dicom_Reader.slice_info])  # z axis coords in the physic coords 
        for i, z in enumerate(zs): # for each slice
            indexes = np.where(coords[:,-1]==z)[0]
            coords[indexes, -1] = i

        # for later use
        self.dose_grid = coords
        self.doseGrid_zz_yy_xx = (coords[:,2], coords[:,1], coords[:,0])

    def _parse_dose_torch(self, vector_dose):
        ''' turn 1D dose to 3D and interp the 3D dose
        return: 3D dose (1, 1, D//2, H=256, W=256) '''
        D, H, W = self.hparam.MCDose_shape 
        doseGrid_shape = self.geometry.doseGrid.size.tolist()[::-1]
        dose = torch.zeros(doseGrid_shape, dtype=torch.float32, device=self.hparam.device)  # 3D dose @ doseGrid size
        dose[self.doseGrid_zz_yy_xx] = vector_dose  # index vector_dose to 3D dose
        dose = torch.nn.functional.interpolate(dose.view([1,1]+doseGrid_shape), size=(D//2,H,W), mode='trilinear', align_corners=False)  # interpolate only support 5D input
        #  dose = torch.nn.functional.interpolate(dose.view(1,1,D,H*2,W*2), size=(D//2,H,W), mode='nearest')
        dose = dose.squeeze()
        return dose

    def get_unit_pencilBeamDose(self, beam_id, segment):
        ''' 
        Arguments: 
            beam_id: int
            segment: ndarray (#validBixels==hxw, )
        Return:
            pbDose: tensor (D=61, H=128, W=128) 
        '''
        pbDose = cal_dose(self.dict_deps[beam_id], segment) # cal unit dose (#dose_grid, )
        pbDose = pbDose[0:self.data.get_pointNum_from_organName('ITV_skin')]  # only consider the skin dose
        pbDose = self._parse_dose_torch(pbDose)  # 3D dose tensor 61x256x256
        pbDose = transforms.CenterCrop(128)(pbDose)  # 61x128x128
        return pbDose

class NeuralDose():
    def __init__(self, hparam, data):
        self.hparam = hparam
        self.data   = data
        self.net    = self.load_net()
        self.PB     = PencilBeam(hparam, data)

        data_dir = Path(hparam.data_dir)
        CTs = np.load(data_dir.joinpath('CTs.npz'))['CTs']  # 3d ndarray (D,H,W) or 4d ndarray (#beam,D,H,W)
        self.CTs = torch.tensor(CTs, dtype=torch.float32, device=hparam.device)
        
        self.pbmcDoses_opened = []
        for fn in list(braceexpand(str(data_dir.joinpath('mcpbDose_{1..6}000000.npz')))):  # these npz are generated from the whole-opened leaf pairs
            self.pbmcDoses_opened.append(load_npz(fn)['mcDose'])  # npz containing multiple keys cannot be accessed in parallel

        self.pbmcDoses_opened = np.stack(self.pbmcDoses_opened, axis=0) # (beam, D, H, W)
        self.pbmcDoses_opened = torch.tensor(self.pbmcDoses_opened, dtype=torch.float32, device=hparam.device)

        self.transform = Transform(hparam)
   
    def get_neuralDose_for_a_beam(self, beam_id, MUs, segs, mask, requires_pencilBeamDose=True):
        '''Arguments:
                MUs:  tensor (#apertures,)
                segs: tensor (#bixels, #apertures), all rays including valid and non-valid rays
                mask: tensor (h,w), hxw==#bixels, bool, 1 indicates valid ray 
           Returns:
                neuralDose: tensor (D/2=61,H=centerCrop128,W=centerCrop128), non-unit neural dose 
                pbDose:     tensor (D/2=61,H=centerCrop128,W=centerCrop128), non-unit pencil beam dose
        '''
        assert isinstance(MUs,  torch.Tensor)
        assert isinstance(segs, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        neuralDose, pbDose = 0, 0
        for i in range(segs.shape[-1]): # for each aperture
            seg = segs[:,i][mask.flatten()]
            unitNeuralDose, unitPBDose = self.get_neuralDose_for_an_aperture(seg, beam_id) # 3D unit dose (D,H,W)
            neuralDose += unitNeuralDose * MUs[i].to('cpu')  # x MU
            if requires_pencilBeamDose:
                pbDose += unitPBDose     * MUs[i].detach().to('cpu')  # x MU
        if requires_pencilBeamDose:
            return neuralDose, pbDose 
        else:
            return neuralDose

    def get_unit_neuralDose_for_a_beam(self, beam_id, seg, mask):
        '''Arguments:
                seg : tensor (#bixels, ), all rays including valid and non-valid rays
                mask: tensor (h,w), hxw==#bixels, bool, 1 indicates valid ray
            Returns:
                pbDose:     tensor (D,H,W) unit pencil beam dose
                neuralDose: tensor (D,H,W) unit neural dose'''
        assert isinstance(seg,  torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        seg = seg[mask.flatten()]
        unitNeuralDose, unitPBDose = self.get_neuralDose_for_an_aperture(seg, beam_id) # 3D unit dose (D,H,W)
        return unitNeuralDose, unitPBDose

    def get_neuralDose_for_an_aperture(self, segment, beam_id):
        ''' Arguments:
                    segment: bool tensor vector (#valid_bixels, )  NOTE: valid rays!
                    beam_id: int
            Returns:
                    unitPBDose:     tensor (D/2,H=centerCrop128,W=centerCrop128), unit dose
                    unitNeuralDose: tensor (D/2,H=centerCrop128,W=centerCrop128), unit dose
        '''
        # PencilBeam dose
        # print(f'{self.dict_deps[beam_id].shape}----{segment.shape}')
        #  unitPBDose = cal_dose(self.dict_deps[beam_id], segment) # cal unit dose (#dose_grid, )
        #  unitPBDose = unitPBDose[0:self.data.get_pointNum_from_organName('ITV_skin')]  # only consider the skin dose
        #  unitPBDose = self.PB._parse_dose_torch(unitPBDose)  # 3D dose 61x256x256
        unitPBDose = self.PB.get_unit_pencilBeamDose(beam_id, segment)  # 3D dose 61x128x128

        # net inputs
        mcdose_opened = self.pbmcDoses_opened[beam_id-1] # get whole-opened mcdose for the current gantry angle
        inputs = torch.stack([self.CTs, mcdose_opened, unitPBDose], dim=0)
        inputs = self.transform(inputs) # D=64,H,W
        inputs = inputs.unsqueeze(0)  # B=1,D=64,H,w

        # forward throught net  # with torch.no_grad():
        with torch.cuda.amp.autocast():
            unitNeuralDose = self.net(inputs.to('cpu'))
        unitNeuralDose = torch.relu(unitNeuralDose.squeeze())
        unitNeuralDose = self.transform.strip_padding_depths(unitNeuralDose) # 1,61,128,128
        return unitNeuralDose, unitPBDose.detach() 

    def load_net(self):
        net = UNet3D(in_channels=3, n_classes=1, norm_type=self.hparam.norm_type)
        cprint(f'loading neuralDose network from {self.hparam.ckpt_path}', 'green')
        ckpt = torch.load(self.hparam.ckpt_path, map_location=torch.device(self.hparam.device))
        state_dict = {}
        for k, v in ckpt['state_dict'].items(): state_dict[k.replace('net.', '')] = v
        missing_keys, unexpected_keys = net.load_state_dict(state_dict)
        cprint('missing_keys', 'green')
        print(missing_keys)
        cprint('unexpected_keys', 'green')
        print(unexpected_keys)

        net.eval()
        for k, v in net.named_parameters(): v.requires_grad_(False)  # try to save memory
        net.to('cpu')
        return net
