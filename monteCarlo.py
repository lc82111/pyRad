#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pdb, glob, re, unittest, shutil, pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace
from orderedbunch import OrderedBunch
from multiprocessing import Process, Manager
from termcolor import colored, cprint
from orderedbunch import OrderedBunch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc
from skimage.transform import resize, rotate 

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from loss import Loss
from utils import *
from data import Data


class MonteCarlo():
    def __init__(self, hparam, data):
        self.hparam = hparam
        self.data = data

        self.nb_leafPairs = 51    # 51 leaf pairs
        self.x_spacing    = 0.5   # cm
        self.nb_beams     = data.num_beams
        cprint(f'MonteCarlo: using number of leaves {self.nb_leafPairs}, grid size {self.x_spacing} cm for x axis', 'red')  # juyao told me
        
        self._get_leafBottomEdgePosition()
        self._get_leafInJawField()  # get y axis leaf position from jaw_y1 ,jaw_y2

    def _get_leafBottomEdgePosition(self):
        '''
        the leaf coords is:     jaw_y2(+)
                            jaw_x1(-)   jaw_x2(+)
                                jaw_y1(-)
        Return: self.coords, list of 51 leaves' bottom edge positions 
        '''
        ## read FM_info file
        FM_info_template = os.path.join(self.hparam.winServer_MonteCarloDir, 'templates', 'FM_info.txt') 
        with open(FM_info_template, 'r') as f:
            lines = f.readlines()

        ## 0. get the thickness of the 51 pair leaves
        is_thick_line = False
        thicks = []
        leaf_num = 0
        for line in lines:
            if 'MLC_LeafThickness' in line:
                is_thick_line = True
                continue
            if leaf_num == self.nb_leafPairs:
                break
            if is_thick_line:
                thicks.append(float(line.replace('\n','')))
                leaf_num += 1
        #print(thicks)
        #print(sum(thicks))
        #print(f'center leaf thickness: {thicks[25]}')

        ## 1. get edge bottom coord of leaves (51 pairs) 
        coords = [] # leaves bottom edges 

        # upper half leaves: total 25 edge bottom positions
        coord26thLeafUp = thicks[25]/2.  # 26-th leaf with its center at y=0
        coords.append(coord26thLeafUp) # +1 position
        for i in range(24, 0, -1): # [24, 0], +24 positions
            coord26thLeafUp += thicks[i]
            coords.append(coord26thLeafUp) 
        coords = coords[::-1]

        # lower half leaves: total 26 edge bottom positions
        coord26thLeafbot = -thicks[25]/2.
        coords.append(coord26thLeafbot) # +1 position
        for i in range(26, self.nb_leafPairs): # [26, 50], +25 positions
            coord26thLeafbot -= thicks[i]
            coords.append(coord26thLeafbot)

        # round to 2 decimals to consistent with TPS 
        self.coords = [round(c, 1) for c in coords]

    def _get_leafInJawField(self):
        '''
        get y axis leaf positions by finding the leaves in jaw field 

        Return: self.dict_jawsPos {beam_id: [x1,x2,y1,y2]}, self.dict_inJaw {beam_id: (51,)}
        '''
        self.dict_jawsPos = OrderedBunch() # jaw positions
        self.dict_inJaw= OrderedBunch()  # bool vector indicate leaves in jaw Filed 
        ## get jaw positions from seg*.txt file
        seg_files = glob.glob(os.path.join(self.hparam.winServer_MonteCarloDir, 'templates',  'Seg_beamID*.txt'))
        seg_files.sort() # sort to be consistent with beam_id

        for beam_id, seg in enumerate(seg_files):
            beam_id += 1
            H, W = self.data.dict_bixelShape[beam_id]
            #  print(f'beam_ID:{beam_id}; file_name:{seg}')
            with open(seg, 'r') as f:
                lines = f.readlines()

            ## get jaw positions
            is_jaw_line = False
            jaw = OrderedBunch()
            for line in lines:
                if 'MU_CollimatorJawX1' in line:
                    is_jaw_line = True
                    continue
                if is_jaw_line:
                    position = line.split(' ')[1:5]
                    position = [float(p) for p in position]
                    jaw.x1, jaw.x2, jaw.y1, jaw.y2 = position
                    print(f'jaw position: {jaw.x1, jaw.x2, jaw.y1, jaw.y2}')
                    break
            self.dict_jawsPos[beam_id] = jaw

            ## Is a leaf in jaws' open field?
            # for upper half leaves: if (leaf bottom edge > jaw_y1) {this leaf in valid field}
            # for lower half leaves: if (leaf upper  edge < jaw_y2) {this leaf in valid field}   
            self.dict_inJaw[beam_id] = np.empty((self.nb_leafPairs,), dtype=np.bool)
            for i, c in enumerate(self.coords):
                in_field = False
                if (c<jaw.y2 and c>jaw.y1): 
                    in_field = True
                if (c<jaw.y2 and self.coords[i-1]>jaw.y1):  # consider upper edge
                    in_field = True
                self.dict_inJaw[beam_id][i] = in_field
                #  print(f'{in_field}---{i}: {c}')
            #  print(f'{self.dict_inJaw[beam_id].sum()}')
            assert self.dict_inJaw[beam_id].sum() == H, f'H={H}, inJaw={self.dict_inJaw[beam_id].sum()}'
            #print(f'H={H}, inJaw={self.dict_inJaw[beam_id].sum()}')

    def _get_x_axis_position(self, dict_segments):
        '''get x axis positions 
         Arguments: 
            dict_segments: {beam_id, (#apertures, h, w)}
         Return: 
            self.nb_apertures: int
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order. '''
        self.dict_lrs = OrderedBunch()  # {beam_id: (#aperture, 51)}

        def get_leafPos_for_a_row(row):
            '''
            [0.0] 0 [0.5] 0 [1.0] 1 [1.5] 1 [2.0] 0 [2.5] 0 [3.0]
            '''
            jaw_x1 = self.dict_jawsPos[beam_id].x1
            if (row==0).all():  # closed row
                lr = default_lr; first,last=0,0 
            else: # opened row
                first, last = np.nonzero(row)[0][[0,-1]]  # get first 1 and last 1 positions
                #  last += 1 # block the left bixel of first 1, and right bixel of last 1; TODO +1?
                l = jaw_x1 + first*self.x_spacing # spacing 0.5mm
                r = jaw_x1 + last *self.x_spacing # spacing 0.5mm
                lr = '{:.2f} {:.2f}\n'.format(l, r)
            #  cprint(f'row:{row_idx}; {first}  {last};  {lr}', 'green')
            return lr

        self.nb_apertures = dict_segments[1].shape[0]
        for beam_id, segs in dict_segments.items():  # 0. for each beam
            pos = self.dict_jawsPos[beam_id].x1-self.x_spacing  # leaf closed at jaw_x1-0.5 by default 
            default_lr = '{:.2f} {:.2f}\n'.format(pos, pos)  # by default, leaves closed 
            self.dict_lrs[beam_id] = np.full((self.nb_apertures, self.nb_leafPairs), default_lr, dtype=object)  # (#aperture, 51), 
            for i in range(self.nb_apertures):   # 1. for each aperture
                row_idx = 0  # 3. row index in jaw, segment only has rows in jaw
                for j in range(self.nb_leafPairs): # 2. for each row of 51 leaves 
                    if self.dict_inJaw[beam_id][j]:
                        lr = get_leafPos_for_a_row(segs[i, row_idx])
                        self.dict_lrs[beam_id][i, j] = lr
                        row_idx += 1
                self.dict_lrs[beam_id][i] = self.dict_lrs[beam_id][i, ::-1]  # NOTE: In TPS, 51 leaf pairs are in reversed order. 

    def write_to_seg_txt(self, seg_dir='Segs'):
        """
        Write Seg_{beam_id}_{aperture_id}.txt to the Segs dir on the shared disk of windowsServer
        Args: 
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order.
            self.nb_apertures
            self.nb_beams
            seg_dir: write segments in the directory of windowsServer
        Outputs:
            seg*.txt on windowsServer 
        """
        cprint(f'write {self.nb_beams*self.nb_apertures} Seg*.txt files to {self.hparam.winServer_MonteCarloDir}/{seg_dir}.', 'green')
        for beam_id in range(1, self.nb_beams+1):
            seg_template = os.path.join(self.hparam.winServer_MonteCarloDir, 'templates', f'Seg_beamID{beam_id}.txt')
            with open(seg_template, 'r') as f:
                lines = f.readlines()
            for aperture_id in range(0, self.nb_apertures):
                ap_lines = lines.copy() + [None]*51
                ap_lines[-51: ] = self.dict_lrs[beam_id][aperture_id] # 51 leaves positions

                # write Seg*.txt 
                save_path = os.path.join(self.hparam.winServer_MonteCarloDir, 'Segs', f'Seg_{beam_id}_{aperture_id}.txt')
                with open(save_path, "w") as f:
                    f.writelines(ap_lines)
                cprint(f'Writing Seg_{beam_id}_{aperture_id}.txt', 'green')

        cprint(f'Done. {self.nb_beams*self.nb_apertures} Seg*.txt files have been written to Dir {self.hparam.winServer_MonteCarloDir}/{seg_dir}.', 'green')

    def get_JY_MCdose(self, dosefilepath, numberOfFractions):
        ''' Return: MCDose computer by ju yao, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        dose = 0
        dosefiles = glob.glob(dosefilepath)
        for dose_file in dosefiles:
            cprint(f'read monteCarlo dose from {dose_file}', 'green')
            with open(dose_file, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                data = data.reshape(*self.hparam.MCDose_shape)
                data = np.swapaxes(data, 2, 1)
                dose += data
        return dose * numberOfFractions

    def cal_unit_MCdose_on_winServer(self, dict_segments):
        ''' call FM.exe and gDPM.exe on windowsServer. 
        Arguments:
            dict_segments: {beam_id, (#apertures, h, w)}
        Return: 
            unitMUDose, ndarray (nb_beams*nb_apertures, D, H, W)  '''
        self._get_x_axis_position(dict_segments)  # get x axis position from the saved random generated fluences
        self.write_to_seg_txt(seg_dir='Segs')

        cprint(f'compute unit MU Dose on winServer and save results to {self.hparam.winServer_MonteCarloDir}', 'green')
        pdb.set_trace()
        call_FM_gDPM_on_windowsServer(self.hparam.patient_ID, self.nb_beams, self.nb_apertures, self.hparam.winServer_nb_threads)
        pdb.set_trace()

    def get_unit_MCdose_from_winServer(self, beam_id, aperture_id): 
        ''' get calculated dose dpm_result_{beam_id}_{aperture_id}Ave.dat from windowsServer.
            Arguments:
                beam_id, aperture_id: int
            Return:
                mcDose: ndarray (D/2=61, H=centerCrop128, W=centerCrop128) == net_output_shape
        '''
        dpm_result_path = Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results', f'dpm_result_{beam_id}_{aperture_id}Ave.dat')
        cprint(f'read monteCarlo unit dose from {dpm_result_path}', 'green')
        with open(dpm_result_path, 'rb') as f:
            dose = np.fromfile(f, dtype=np.float32)
            dose = dose.reshape(*self.hparam.MCDose_shape)
        mcDose = np.swapaxes(dose, 2, 1)

        D, H, W = self.hparam.MCDose_shape 
        assert mcDose.shape == (D,H,W)
         
        mcDose = resize(mcDose, (D//2,H,W), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False)
        mcDose = np.where(mcDose<0, 0, mcDose).astype(np.float32)  # bicubic(order=3) resize may create negative values
        mcDose = torch.tensor(mcDose, dtype=torch.float32, device=self.hparam.device)
        mcDose = transforms.CenterCrop(size=(128,128))(mcDose)
        mcDose = to_np(mcDose)
        assert mcDose.shape == self.hparam.net_output_shape

        return mcDose
