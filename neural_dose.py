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

import os, pdb, sys, pickle, time
from argparse import ArgumentParser
from termcolor import colored, cprint
from io import StringIO
from pathlib import Path
import pandas as pd

from neuralDose.utils import MyModelCheckpoint
from utils import *
from data import Geometry 
from neuralDose.net.Unet3D import UNet3D 
from neuralDose.data.datamodule import Transform 


class NeuralDose():
    def __init__(self, hparam, data, PB):
        self.times  = []
        self.hparam = hparam
        self.data   = data
        self.PB     = PB
        self.net    = self.load_net()
        self.is_first = True

        data_dir = Path(hparam.data_dir)
        CTs = np.load(data_dir.joinpath('CTs.npz'))['CTs']  # 3d ndarray (D,H,W) or 4d ndarray (#beam,D,H,W)
        self.CTs = torch.tensor(CTs, dtype=torch.float32, device=hparam.device)
        
        self.pbmcDoses_opened = []
        for fn in list(braceexpand(str(data_dir.joinpath('mcpbDose_{1..%s}000000.npz'%data.num_beams)))):  # these npz are generated from the whole-opened leaf pairs
            self.pbmcDoses_opened.append(load_npz(fn)['mcDose'])  # npz containing multiple keys cannot be accessed in parallel

        self.pbmcDoses_opened = np.stack(self.pbmcDoses_opened, axis=0) # (beam, D, H, W)
        self.pbmcDoses_opened = torch.tensor(self.pbmcDoses_opened, dtype=torch.float32, device=hparam.device)

        self.transform = Transform(hparam)
        
        self.cachedUnitNeuralDose = {}
        for i in range(1, data.num_beams+1):
            self.cachedUnitNeuralDose[i] = []

    def get_neuralDose_for_a_beam(self, beam_id, MUs, segs, mask):
        '''Arguments:
                MUs:  tensor (#apertures,)
                segs: tensor (#bixels, #apertures), all rays including valid and non-valid rays
                mask: tensor (h,w), hxw==#bixels, bool, 1 indicates valid ray 
           Returns:
                neuralDose: tensor (D/2=61,H=centerCrop128,W=centerCrop128), non-unit neural dose 
                pbDose:     tensor (D/2=61,H=centerCrop128,W=centerCrop128), non-unit pencil beam dose
        '''
        if isinstance(beam_id, torch.Tensor):
            beam_id = int(to_np(beam_id))

        neuralDose = 0 
        if self.is_first and self.hparam.not_use_apertureRefine:
            self.cachedUnitNeuralDose[beam_id] = {}

        def _get_unitDose():
            seg = segs[:,i][mask.flatten()]
            return self.get_neuralDose_for_an_aperture(seg, beam_id) # 3D unit dose (D,H,W)

        for i in range(segs.shape[-1]): # for each aperture
            if self.is_first and self.hparam.not_use_apertureRefine:
                unitDose    = _get_unitDose()  # 3D dose 61x128x128
                neuralDose += unitDose * MUs[i]  # x MU
                self.cachedUnitNeuralDose[beam_id][i] = unitDose.detach()
            elif not self.is_first and self.hparam.not_use_apertureRefine:
                neualDose += self.cachedUnitNeuralDose[beam_id][i] * MUs[i]
            else:  # first and AR, not first and AR
                unitDose   = _get_unitDose()  # 3D dose 61x128x128
                neuralDose+= unitDose * MUs[i]  # x MU

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
        unitNeuralDose = self.get_neuralDose_for_an_aperture(seg, beam_id) # 3D unit dose (D,H,W)
        return unitNeuralDose 

    def get_neuralDose_for_an_aperture(self, segment, beam_id):
        ''' Arguments:
                    segment: bool tensor vector (#valid_bixels, )  NOTE: valid rays!
                    beam_id: int
            Returns:
                    unitPBDose:     tensor (D/2,H=centerCrop128,W=centerCrop128), unit dose
                    unitNeuralDose: tensor (D/2,H=centerCrop128,W=centerCrop128), unit dose
        '''
        # unitPBDose
        unitPBDose = self.PB.get_unit_pencilBeamDose(beam_id, segment)  # 3D dose 61x128x128

        # net inputs
        mcdose_opened = self.pbmcDoses_opened[beam_id-1] # get whole-opened mcdose for the current gantry angle
        inputs = torch.stack([self.CTs, mcdose_opened, unitPBDose], dim=0)
        inputs = self.transform(inputs) # D=64,H,W
        inputs = inputs.unsqueeze(0)  # B=1,D=64,H,w

        # forward throught net  # with torch.no_grad():
        with torch.cuda.amp.autocast():
            unitNeuralDose = self.net(inputs)
        unitNeuralDose = torch.relu(unitNeuralDose.squeeze())
        unitNeuralDose = self.transform.strip_padding_depths(unitNeuralDose) # 1,61,128,128
        return unitNeuralDose

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
        #net.to('cpu')
        net.to(self.hparam.device)
        return net
