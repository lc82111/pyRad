#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pdb, glob, re, unittest, shutil, pickle
from argparse import ArgumentParser, Namespace
from orderedbunch import OrderedBunch
from multiprocessing import Process, Manager
from termcolor import colored, cprint
from data import Data
from orderedbunch import OrderedBunch
import numpy as np
from socket import *

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc

import torch
from torch.utils.tensorboard import SummaryWriter
from loss import Loss
from utils import *


class MonteCarlo():
    def __init__(self, hparam, data):
        self.hparam = hparam
        self.data = data

        self.nb_leafPairs = 51    # 51 leaf pairs
        self.x_spacing    = 0.5   # cm
        self.nb_beams     = data.num_beams
        cprint(f'using number of leaves {self.nb_leafPairs}, grid size {self.x_spacing} cm for x axis', 'red')  # juyao told me
        
        self._get_leafBottomEdgePosition()
        self._get_leafInJawField()  # get y axis leaf position from jaw_y1 ,jaw_y2

    def get_random_apertures(self, nb_apertures=1000):
        ''' generate random apertures for deep learning training set. 
            Arguments: nb_apertures we will generate this number random apertures
            Return: self.dict_randomApertures {beam_id: ndarray(nb_apertures, H, W)} '''
        def get_random_shape(H,W):
            if np.random.randint(0,2):
                img = random_shapes((H, W), max_shapes=3, multichannel=False, min_size=min(H,W)//3, allow_overlap=True, intensity_range=(1,1))[0]
                img = np.where(img==255, 0, img)
            else:
                img = np.zeros((H,W), dtype=np.uint8)
                for i in range(len(img)):  # for each row
                    l, r = np.random.randint(0, W+1, (2,))
                    if l==r: continue
                    if l>r: l,r = r,l 
                    img[i, l:r] = 1
            return img
        
        self.nb_apertures = nb_apertures 

        save_path = Path(hparam.patient_ID).joinpath('dataset/dict_randomApertures.pickle')
        if os.path.isfile(save_path):
            self.dict_randomApertures = unpickle_object(save_path)
            return

        self.dict_randomApertures = OrderedBunch() 
        for beam_id in range(1, self.nb_beams+1):  # for each beam
            H, W = self.data.dict_rayBoolMat[beam_id].shape
            self.dict_randomApertures[beam_id] = np.zeros((self.nb_apertures, H, W), np.uint8)  # default closed apertures
            for i, apt in enumerate(self.dict_randomApertures[beam_id]):  # for each apterture 
                if i==0:   # skip first aperture for each beam to get a all-leaf-opened aperture
                    self.dict_randomApertures[beam_id][i] = np.ones((H,W), np.uint8)
                else:
                    self.dict_randomApertures[beam_id][i] = get_random_shape(H,W)
        pickle_object(save_path, self.dict_randomApertures)

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
        self.coords = [round(c, 2) for c in coords]

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
            H, W = self.data.dict_rayBoolMat[beam_id].shape
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

    def _get_x_axis_position(self, flag):
        '''get x axis positions from optimized_segments_MUs_file or self.dict_randomApertures.
         Arguments: 
            flag: randomApertures|optimized_segments_MUs
         Return: 
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order. '''
        self.dict_lrs = OrderedBunch()  # {beam_id: (#aperture, H)}

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

        if flag == 'randomApertures':
            for beam_id, apts in self.dict_randomApertures.items():  # 0. for each beam
                #  print(f'\n beam_id:{beam_id}')
                H, W = self.data.dict_rayBoolMat[beam_id].shape
                #  print(f'height:{H}; width:{W}')

                pos = self.dict_jawsPos[beam_id].x1-self.x_spacing  # leaf closed at jaw_x1-0.5 by default 
                default_lr = '{:.2f} {:.2f}\n'.format(pos, pos)  # by default, leaves closed 
                self.dict_lrs[beam_id] = np.full((self.nb_apertures, self.nb_leafPairs), default_lr, dtype=object)  # (#aperture, 51), 
                for a in range(self.nb_apertures):   # 1. for each aperture
                    row_idx = 0
                    for i in range(self.nb_leafPairs): # 2. for each row
                        if self.dict_inJaw[beam_id][i]:
                            lr = get_leafPos_for_a_row(apts[a, row_idx])
                            self.dict_lrs[beam_id][a, i] = lr
                            row_idx += 1
                    self.dict_lrs[beam_id][a] = self.dict_lrs[beam_id][a, ::-1]  # NOTE: In TPS, 51 leaf pairs are in reversed order. 

        elif flag == 'optimized_segments_MUs':
            file_name = os.path.join(self.hparam.optimized_segments_MUs_file, 'optimized_segments_MUs.pickle')
            with open(file_name, 'rb') as f:
                self.segs_mus = pickle.load(f)
            self.nb_apertures = len(self.segs_mus[1]['MU']) 
            self.old_MUs      = np.empty((self.nb_beams*self.nb_apertures, 1, 1, 1), dtype=np.float32)
            assert self.nb_beams == len(self.segs_mus)

            for beam_id, seg_mu in self.segs_mus.items():  # 0. for each beam
                #  print(f'\n beam_id:{beam_id}')
                H, W = self.data.dict_rayBoolMat[beam_id].shape
                #  print(f'height:{H}; width:{W}')
                segs, mus = seg_mu['Seg'], seg_mu['MU']
                self.old_MUs[(beam_id-1)*self.nb_apertures: (beam_id-1)*self.nb_apertures+self.nb_apertures] = mus.reshape((self.nb_apertures,1,1,1)) 

                pos = self.dict_jawsPos[beam_id].x1-self.x_spacing  # leaf closed at jaw_x1-0.5 by default 
                default_lr = '{:.2f} {:.2f}\n'.format(pos, pos)  # by default, leaves closed 
                self.dict_lrs[beam_id] = np.full((self.nb_apertures, self.nb_leafPairs), default_lr, dtype=object)  # (#aperture, 51), 
                for aperture in range(self.nb_apertures):   # 1. for each aperture
                    seg = segs[:, aperture]
                    seg = seg.reshape(H,W)
                    row_idx = 0
                    for i in range(self.nb_leafPairs): # 2. for each row
                        if self.dict_inJaw[beam_id][i]:
                            lr = get_leafPos_for_a_row(seg[row_idx])
                            self.dict_lrs[beam_id][aperture, i] = lr
                            row_idx += 1
                    self.dict_lrs[beam_id][aperture] = self.dict_lrs[beam_id][aperture, ::-1]  # NOTE: In TPS, 51 leaf pairs are in reversed order. 

    def write_to_seg_txt(self, seg_dir='Segs'):
        """
        Write seg*.txt to the shared disk of windowsServer
        Args: 
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order.
            self.nb_apertures
            self.nb_beams
            seg_dir: write segments in the directory of windowsServer
        Outputs:
            seg*.txt on windowsServer 
        """
        ## write Seg_{beam_id}_{aperture_id}.txt 
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

    def cal_unit_MCdose_on_windows(self, flag):
        ''' call FM.exe and gDPM.exe on windowsServer. 
        Arguments:
            flag: randomApertures|optimized_segments_MUs
        Return: 
            unitMUDose, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        self._get_x_axis_position(flag)  # get x axis position from the saved random generated fluences
        self.write_to_seg_txt(seg_dir='Segs')

        cprint(f'compute unit MU Dose on winServer and save results to {self.hparam.winServer_MonteCarloDir}', 'green')
        pdb.set_trace()
        call_FM_gDPM_on_windowsServer(self.hparam.patient_ID, self.nb_beams, self.nb_apertures, hparam.winServer_nb_threads)
        pdb.set_trace()

    def get_unit_MCdose(self, uid): 
        ''' get calculated dose dpm_result_{beam_id}_{aperture_id}Ave.dat from windowsServer.
            Arguments:
                uid: {beam_id}_{aperture_id}
            Return:
                mcDose(#slice, H, W) '''
        dpm_result_path = Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results', f'dpm_result_{uid}Ave.dat')
        cprint(f'read monteCarlo unit dose from {dpm_result_path}', 'green')
        with open(dpm_result_path, 'rb') as f:
            dose = np.fromfile(f, dtype=np.float32)
            dose = dose.reshape(*hparam.MCDose_shape)
        mcDose = np.swapaxes(dose, 2, 1)
        return mcDose
    
    def get_all_beams_unit_MCdose(self):
        ''' Return: unitMUDose, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        pdb.set_trace()
        if os.path.isfile(self.hparam.unitMUDose_npz_file):
            cprint(f'load {self.hparam.unitMUDose_npz_file}', 'green')
            return np.load(self.hparam.unitMUDose_npz_file)['npz']
        else:
            if not Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results', 'dpm_result_1_0Ave.data').is_file(): 
                self.cal_unit_MCdose_on_windows(flag='optimized_segments_MUs')
            cprint(f'fetching gDPM results from winServer and save them to local disk: {self.hparam.unitMUDose_npz_file}', 'green')
            unitMUDose = np.empty((self.nb_beams*self.nb_apertures, ) + self.hparam.MCDose_shape, dtype=np.float32)
            idx = 0
            for beam_id in range(1, self.nb_beams+1):
                for aperture_id in range(0, self.nb_apertures):
                    unitMUDose[idx] = self.get_unit_MCdose(uid=f'{beam_id}_{aperture_id}')
                    idx += 1

            np.savez(self.hparam.unitMUDose_npz_file, npz=unitMUDose)
            return unitMUDose
