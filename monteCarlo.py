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

def call_FM_gDPM_on_windowsServer(PID, nb_beams, nb_apertures, nb_threads):
    cprint(f'send msg to windows Server to call FM.exe and gDPM.exe.', 'green')

    host = "192.168.0.125" # set to IP address of target computer
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    msg = "seg.txt files are ready;%s;%s;%s;%s"%(PID, nb_beams, nb_apertures, nb_threads)
    data = msg.encode('utf-8')
    UDPSock.sendto(data, addr)
    UDPSock.close()
    cprint("messages send.", 'green')

    try:
        host = "0.0.0.0"
        port = 13001
        buf = 1024
        addr = (host, port)
        UDPSock = socket(AF_INET, SOCK_DGRAM)
        UDPSock.bind(addr)
        cprint("Waiting to receive messages...", 'green')
        while True:
            (data, addr) = UDPSock.recvfrom(buf)
            msg = '%s'%(data)
            if 'done' in msg:
                break
        UDPSock.close()
        cprint("Receive compute done from winServer.", 'green')
    except KeyboardInterrupt:
        UDPSock.close()
        os._exit(0)
    except:
        cprint("Error in call_FM_gDPM_on_windowsServer", 'red')
        os._exit(0)

def parse_MonteCarlo_dose(MCDose, data):
    ''' Return: dict_organ_dose {organ_name: dose ndarray (#organ_dose, )} '''
    dict_organ_dose = OrderedBunch()
    for organ_name, msk in data.organ_masks.items():
        dict_organ_dose[organ_name] = MCDose[msk]
    return dict_organ_dose 

class MonteCarlo():
    def __init__(self, hparam, data):
        self.hparam = hparam
        self.data = data

        self.nb_leafPairs = 51   # 51 leaf pairs
        self.x_spacing    = 0.5  # mm
        self.nb_threads   = 8    # threads windows server will run

        self._get_leafBottomEdgePosition()
        self._get_leafInJawField()  # get y axis leaf position from jaw_y1 ,jaw_y2
        self._get_x_axis_position()  # get x axis position from optimized_segments_MUs_file

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
        self.coords = coords

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
                    #print(f'jaw position: {jaw.x1, jaw.x2, jaw.y1, jaw.y2}')
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

    def _get_x_axis_position(self):
        '''
         get x axis position from optimized_segments_MUs_file
         Return: 
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order.
            self.nb_beams
            self.nb_apertures
        '''
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
        file_name = os.path.join(self.hparam.optimized_segments_MUs_file, 'optimized_segments_MUs.pickle')
        with open(file_name, 'rb') as f:
            self.segs_mus = pickle.load(f)
        self.nb_apertures = len(self.segs_mus[1]['MU'])
        self.nb_beams     = len(self.segs_mus)
        self.old_MUs      = np.empty((self.nb_beams*self.nb_apertures, 1, 1, 1), dtype=np.float32)

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

    def write_to_seg_txt(self):
        """
        Write seg*.txt to the shared disk of windowsServer
        Args: 
            self.dict_lrs {beam_id: strings (#aperture, 51)}, NOTE: 51 leaf pairs in reversed order.
            self.nb_apertures
            self.nb_beams
        Outputs:
            seg*.txt 
        """
        ## write Seg_{beam_id}_{aperture_id}.txt 
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

        cprint(f'Done. {self.nb_beams*self.nb_apertures} Seg*.txt files have been written to Dir {self.hparam.winServer_MonteCarloDir}/segs.', 'green')

    def get_JY_MCdose(self, dosefilepath):
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
        return dose * 33 # 33 is numberOfFractions 

    def get_unit_MCdose(self):
        ''' Return: unitMUDose, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        if os.path.isfile(self.hparam.unitMUDose_npz_file):
            cprint(f'load {self.hparam.unitMUDose_npz_file}', 'green')
            return np.load(self.hparam.unitMUDose_npz_file)['npz']
        else:
            if self.hparam.Calculate_MC_unit_doses:
                cprint(f'compute unit MU Dose on winServer and save results to {self.hparam.winServer_MonteCarloDir}', 'green')
                pdb.set_trace()
                self.write_to_seg_txt()
                call_FM_gDPM_on_windowsServer(self.hparam.patient_ID, self.nb_beams, self.nb_apertures, hparam.winServer_nb_threads)
                pdb.set_trace()

            cprint(f'fetching gDPM results from winServer and save them to local disk: {self.hparam.unitMUDose_npz_file}', 'green')
            unitMUDose = np.empty([self.nb_beams*self.nb_apertures,] + self.hparam.MCDose_shape, dtype=np.float32)
            idx = 0
            for beam_id in range(1, self.nb_beams+1):
                for aperture_id in range(0, self.nb_apertures):
                    dose_file = os.path.join(self.hparam.winServer_MonteCarloDir, 'gDPM_results', f'dpm_result_{beam_id}_{aperture_id}Ave.dat')
                    cprint(f'read monteCarlo unit MU dose from {dose_file}', 'green')
                    with open(dose_file, 'rb') as f:
                        data = np.fromfile(f, dtype=np.float32)
                        dose = data.reshape(*self.hparam.MCDose_shape)
                        dose = np.swapaxes(dose, 2, 1)
                        unitMUDose[idx] = dose
                        idx += 1
            pdb.set_trace()
            if not os.path.isdir(os.path.dirname(self.hparam.unitMUDose_npz_file)):
                os.makedirs(os.path.dirname(self.hparam.unitMUDose_npz_file))
            np.savez(self.hparam.unitMUDose_npz_file, npz=unitMUDose)
            cprint(f'unit MC doses have been saved to local disk: {self.hparam.unitMUDose_npz_file}', 'green')
            return unitMUDose

    def get_unit_MCdose_multiThreads(self, nb_threads=15):
        ''' 
        NOTE: this function is deprecated!!! this function results random unitMUDose, so it should not be used.
        Return: unitMUDose, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        def child(fn, list_unitMUDose):
            #  dose_file = os.path.join(self.hparam.winServer_MonteCarloDir, 'gDPM_results', fn)
            dose_file = os.path.join('./data/MonteCarlo/tmp', fn)
            cprint(f'read monteCarlo unit MU dose from {dose_file}', 'green')
            with open(dose_file, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                dose = data.reshape(*self.hparam.MCDose_shape)
                dose = np.swapaxes(dose, 2, 1)
                list_unitMUDose.append(dose)
        cprint('read unit MCDose from local disk in parallel.', 'green')
        fns = []
        for beam_id in range(1, self.nb_beams+1):
            for aperture_id in range(0, self.nb_apertures):
                fn = f'dpm_result_{beam_id}_{aperture_id}Ave.dat'
                fns.append(fn)

        with Manager() as manager:
            list_unitMUDose = manager.list() 
            for batch_fn in batch(fns, nb_threads):
                ps = []
                for fn in batch_fn:
                    ps.append(Process(target=child, args=(fn, list_unitMUDose)))
                    ps[-1].start()
                for p in ps:
                    p.join()
            unitMUDose = np.asarray(list_unitMUDose)
        return unitMUDose

    def test_MCDose_CT_overlap(self):
        import SimpleITK as sitk
        from skimage.transform import resize

        Msk  = self.data.organ_masks['Parotid_L']

        unitMUDose = self.get_unit_MCdose_multiThreads()
        MUs = np.abs(self.old_MUs) / self.hparam.dose_scale  # x1000
        MCdoses = unitMUDose * MUs
        MCdoses = MCdoses.sum(axis=0, keepdims=False)  #  (#slice, H, W) 
        #  Doses   = MCdoses * Msk
        Doses   = MCdoses

        CT = sitk.GetArrayFromImage(self.data.CT)
        CT = resize(CT, self.hparam.MCDose_shape, order=0, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False)

        pdb.set_trace()

        for i, (ct, dose, msk) in enumerate(zip(CT, Doses, Msk)): 
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(ct, cmap='gray', interpolation='none')
            plt.imshow(msk, cmap='jet', alpha=0.5, interpolation='none')
            plt.subplot(1,2,2)
            plt.imshow(ct, cmap='gray', interpolation='none')
            plt.imshow(dose, cmap='jet', alpha=0.5, interpolation='none')
            plt.savefig(f'test/imgs/{i}.png')
            plt.close()

def get_parameters():
    parser = ArgumentParser()
   
    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)

    # Monte Carlo parameters 
    parser.add_argument("--Calculate_MC_unit_doses", action='store_true', help='if true, running FM.exe and gDPM.exe on winServer')
    parser.add_argument('--patient_ID', default='Pa14Plan53Rx53GPU_2', type=str, help='gDPM.exe will read and save data to this dir', required=True)
    parser.add_argument('--winServer_nb_threads', default=5, type=int, help='number of threads used for gDPM.exe. Too many threads may overflow the GPU memory')

    # misc parameters
    hparam, _ = parser.parse_known_args()
    parser.add_argument('--optimized_segments_MUs_file', default='./results/'+hparam.exp_name, type=str, help='use segments in this file to cal mc dose')
    parser.add_argument('--unitMUDose_npz_file', default='./dataset/MonteCarlo/'+hparam.patient_ID+'/unitMUDose.npz', type=str, help='MCDose fetched from winServer will be save in this file')
    parser.add_argument('--winServer_MonteCarloDir', default='/mnt/win_share/'+hparam.patient_ID, type=str, help='gDPM.exe save MCDose into this directory; this directory is shared by winServer')
    hparam, _ = parser.parse_known_args()
    
    # tps data parameters
    from data import get_parameters as data_get_parameters 
    data_hparam = data_get_parameters() 
    hparam = Namespace(**vars(hparam), **vars(data_hparam))

    return hparam


if __name__ == "__main__":
    hparam = get_parameters()
    data = Data(hparam)
    mc = MonteCarlo(hparam, data)
    mc.get_unit_MCdose()
