#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, pdb, glob, re, unittest, shutil, pickle
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
        cprint("done", 'green')
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
        self._get_leafInJawField()   # get y axis leaf position from jaw_y1 ,jaw_y2
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
                self.mc.write_to_seg_txt()
                call_FM_gDPM_on_windowsServer(self.hparam.patient_ID, self.nb_beams, self.nb_apertures, hparam.winServer_nb_threads)
                pdb.set_trace()

            cprint(f'fetching gDPM results from winServer and save them to local disk: {self.hparam.unitMUDose_npz_file}', 'green')
            unitMUDose = np.empty((self.nb_beams*self.nb_apertures, ) + self.hparam.MCDose_shape, dtype=np.float32)
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

            np.savez(self.hparam.unitMUDose_npz_file, npz=unitMUDose)
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

def test(hparam):
    hparam = hparam
    data = Data(hparam)
    mc = MonteCarlo(hparam, data)
    #  mc.test_MCDose_CT_overlap()
    cprint('done', 'green')


class Optimization():
    def __init__(self, hparam, MonteCarlo, data, loss):
        '''
        MCDose: MonteCarlo dose of 1 MU 
        '''
        self.hparam       = hparam
        self.mc           = MonteCarlo
        self.nb_apertures = MonteCarlo.nb_apertures
        self.nb_beams     = MonteCarlo.nb_beams
        self.data         = data
        self.loss         = loss
        self.step         = 0

        # create a tensorboard summary writer using the specified folder name.
        if hparam.logs_interval != None:
            self.tb_writer = SummaryWriter(hparam.tensorboard_log)
            tmp_hparam = self.hparam.copy()
            for k, v in tmp_hparam.copy().items():
                if type(v) not in [float, str, int, bool ]:
                    #cprint(f'[CongL warning:] tensorboard add_hparams() gets {type(v)} which should be one of int, float, str, bool, or torch.Tensor', 'red')
                    tmp_hparam.pop(k)
            self.tb_writer.add_hparams(dict(tmp_hparam), {})  # save all hyperparamters

        torch.random.manual_seed(0)
        np.set_printoptions(precision=4, sign=' ')

    def _get_optimizer(self, opt_var, lr, steps, optimizer_name, scheduler_name):
        '''ref: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate'''
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(opt_var, lr=lr)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(opt_var, lr=lr, weight_decay=0.01)
        elif optimizer_name == 'adamw_amsgrad':
            optimizer = torch.optim.AdamW(opt_var, lr=lr, weight_decay=0.01, amsgrad=True)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(opt_var, lr=lr)
        else:
            raise NotImplementedError

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                       'min',
                       patience=optimizer_params['patience'],
                       verbose=True,
                       threshold=1e-4,
                       min_lr=optimizer_params['min_lr']) 
        elif scheduler_name == 'CyclicLR': 
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                       base_lr=optimizer_params['base_lr'],
                       max_lr=optimizer_params['max_lr'],
                       step_size_up=optimizer_params['step_size_up'],
                       step_size_down=None, mode='triangular')
        elif scheduler_name == 'CosineAnnealingLR': 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        else:
            raise NotImplementedError

        return optimizer, scheduler

    def run(self, unitMUDose, learning_rate, steps, optimizer_name, scheduler_name):
        # set up optimization vars
        #  MUs = torch.rand((self.nb_beams*self.nb_apertures, 1, 1, 1), dtype=torch.float32, device=self.hparam.device, requires_grad=True) # [0, 1]
        MUs        = torch.tensor(self.mc.old_MUs, dtype=torch.float32, device=self.hparam.device, requires_grad=True) # [0, 1]
        unitMCDose = torch.tensor(unitMUDose, dtype=torch.float32, device=self.hparam.device)

        # optimizer
        optimizer, scheduler = self._get_optimizer([MUs], learning_rate, steps, optimizer_name, scheduler_name)

        # loop
        min_loss, patience = np.inf, 0
        for i in range(steps):
            # forward
            self.step += 1
            loss = self.forward(unitMCDose, MUs)

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=False) # acccumulate gradients

            # best state
            if to_np(loss) < min_loss:
                min_loss = to_np(loss)
                patience = 0
                best_MUs = to_np(MUs)

            # optim
            optimizer.step() # do gradient decent w.r.t MU and partialExp
            scheduler.step() # adjust learning rate

            # early stop
            if to_np(loss) > min_loss:
                patience += 1
            if patience > self.hparam.plateau_patience:
                cprint(f'Loss dose not drop in last {patience} iters. Early stopped.', 'yellow')
                break

        cprint(f'optimization done.\n Min_loss={min_loss}', 'green')
        del unitMUDose
        return best_MUs

    def forward(self, unitMUDose, MUs):
        '''
        1. compute dose from unitMCDose*MUs.
        2. compute loss from dose
        '''
        MUs = torch.abs(MUs) # nonnegative constraint 
        MCdoses = torch.mul(unitMUDose, MUs) #  (#beams*#apertures, #slice, H, W)
        MCdoses = MCdoses.sum(dim=0, keepdim=False)  #  (#slice, H, W) 
        dict_organ_dose = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        loss = self.loss_func(dict_organ_dose, MUs) # (1,)
        return loss

    def loss_func(self, dict_organ_dose, MUs):
        # cal loss
        loss, breaking_points_nums = self.loss.loss_func(dict_organ_dose)

        # tensorboard logs 
        self.tb_writer.add_scalar('loss/total_loss', loss, self.step)
        self.tb_writer.add_scalars('BreakPoints', breaking_points_nums, self.step)
        if self.step % self.hparam.logs_interval == 0:
            print(f"\n step={self.step}. ------------------------------------------  ")
            # print breaking_points_nums
            print('breaking points #: ', end='')
            for organ_name, breaking_points_num in breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            # MU hist
            self.tb_writer.add_histogram(f'MUs_hist', torch.abs(MUs), self.step)  # nonnegative constraint
            # dose histogram
            for organ_name, dose in dict_organ_dose.items():
                if dose.size(0) != 0:
                    self.tb_writer.add_histogram(f'dose histogram/{organ_name}', dose, self.step)
            print("\n loss={:.6f} \n".format(to_np(loss)))

        return loss

def optim(hparam):
    if not os.path.isdir(hparam.refined_segments_MUs_file):
        os.mkdir(hparam.refined_segments_MUs_file)
    del_fold(hparam.tensorboard_log)  # clear log dir, avoid the mess

    data = Data(hparam)

    loss = Loss(hparam, data.csv_table)

    mc = MonteCarlo(hparam, data)
    unitMUDose = mc.get_unit_MCdose()

    optim = Optimization(hparam, mc, data, loss)
    optimized_MUs = optim.run(unitMUDose, hparam.learning_rate, hparam.steps, hparam.optimizer_name, hparam.scheduler_name)

    # save optimized MUs
    save_path = os.path.join(hparam.refined_segments_MUs_file, 'optimized_MUs.pickle')
    pickle_object(save_path, optimized_MUs)

    # release memory
    torch.cuda.empty_cache()

    cprint('all done!!!', 'green')


class Evaluation():
    def __init__(self, hparam):
        self.hparam = hparam

        # init data and loss
        self.data = Data(hparam)
        self.loss = Loss(hparam, self.data.csv_table)

        # deposition matrix (#voxels, #bixels)
        self.deposition = torch.tensor(self.data.deposition, dtype=torch.float32, device=hparam.device)
        
        # MC dose
        self.mc = MonteCarlo(hparam, self.data)
        self.unitMUDose = self.mc.get_unit_MCdose()

    def load_MonteCarlo_organ_dose(self, MUs, name, scale=1):
        MUs = np.abs(MUs) / self.hparam.dose_scale  # x1000
        MCdoses = self.unitMUDose * MUs * scale
        MCdoses = MCdoses.sum(axis=0, keepdims=False)  #  (#slice, H, W) 
        MCdoses = torch.tensor(MCdoses, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'dose':dict_organ_doses, 'name':name})

    def load_JYMonteCarlo_organ_dose(self, name, dosefilepath, scale=1):
        MCdoses = self.mc.get_JY_MCdose(dosefilepath) * scale
        MCdoses = torch.tensor(MCdoses, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'dose':dict_organ_doses, 'name':name})

    def load_Depos_organ_dose(self, name):
        # get seg and MU
        file_name = self.hparam.optimized_segments_MUs_file+'/optimized_segments_MUs.pickle'
        if not os.path.isfile(file_name): raise ValueError(f'file not exist: {file_name}')
        cprint(f'load segments and MUs from {file_name}', 'yellow')
        segments_and_MUs = unpickle_object(file_name)
        dict_segments, dict_MUs = OrderedBunch(), OrderedBunch()
        for beam_id, seg_MU in segments_and_MUs.items():
            dict_segments[beam_id] = torch.tensor(seg_MU['Seg'], dtype=torch.float32, device=self.hparam.device)
            dict_MUs[beam_id]      = torch.tensor(seg_MU['MU'],  dtype=torch.float32, device=self.hparam.device, requires_grad=True)

        # compute fluence
        fluence, _ = computer_fluence(self.data, dict_segments, dict_MUs)
        fluence    = fluence / self.hparam.dose_scale # * 1000
        dict_FMs   = self.data.project_to_fluenceMaps(to_np(fluence))

        # compute dose
        doses = torch.matmul(self.deposition, fluence) # cal dose
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organ_inf)
        
        return OrderedBunch({'fluence':to_np(fluence), 'dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def load_fluence_dose_from_TPS(self, tps_ray_inten_file='./data/TPSray.txt'):
        # intensity
        fluence = np.loadtxt(tps_ray_inten_file)
        fluence = torch.tensor(fluence, dtype=torch.float32, device=self.hparam.device)
        dict_FMs = self.data.project_to_fluenceMaps(to_np(fluence))

        # dose
        doses = torch.matmul(self.deposition, fluence) # cal dose
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organ_inf)

        return OrderedBunch({'fluence':to_np(fluence), 'dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': 'TPS'})

    def comparison_plots(self, plans):
        '''
        plans: list of plan
        '''
        # pop unnessneary organ dose
        organ_filter = ['PTV1-nd2-nx2', 'PTV2', 'PGTVnd', 'PGTVnx', 'Temp.lobe_L', 'Temp.lobe_R', 'Parotid_L', 'Parotid_R', 'Mandible', 'Brainstem+2mmPRV', 'Cord+5mmPRV']
        for plan in plans:
            for name in plan.dose.copy().keys(): 
                if name not in organ_filter: 
                    plan.dose.pop(name)
                    print(f'pop {name}')

        # print loss
        for plan in plans: 
            dict_organ_doses = plan.dose.copy()
            for name, dose in dict_organ_doses.copy().items():
                dict_organ_doses.update({name: dose*self.hparam.dose_scale})  # /1000
            plan_loss, plan_breaking_points_nums = self.loss.loss_func(dict_organ_doses)
            print(f'{plan.name} breaking points #: ', end='')
            for organ_name, breaking_points_num in plan_breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            print(f'losses: loss={plan_loss}\n\n')

        # plot DVH
        fig, ax = plt.subplots(figsize=(20, 10))
        max_dose = 12000
        organ_names = list(plans[0].dose.keys())
        colors = cm.jet(np.linspace(0,1,len(organ_names)))
        #  linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        linestyles = ['solid', 'dashed', 'dashdot'] 
        if len(plans) > 4: raise NotImplementedError

        for i, organ_name in enumerate(organ_names):
            if self.data.organ_inf[organ_name] != 0:
                for pi, plan in enumerate(plans):
                    n, bins, patches = ax.hist(to_np(plan.dose[organ_name]),
                       bins=12000, 
                       linestyle=linestyles[pi], color=colors[i],
                       range=(0, max_dose),
                       density=True, histtype='step',
                       cumulative=-1, 
                       label=f'{plan.name}_{organ_name}_maxDose{int(to_np(plan.dose[organ_name].max()))}')

        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05,1.0))
        ax.set_title('Dose volume Histograms')
        ax.set_xlabel('Absolute Dose cGy')
        ax.set_ylabel('Relative Volume %')
        plt.tight_layout()
        plt.savefig('./tmp.pdf')
        #plt.show()

    def run(self):
        #  CG_Depos_plan        = self.load_Depos_organ_dose('CG_depos')
        #  tps_plan          = self.load_fluence_dose_from_TPS()
        CG_MC_plan           = self.load_MonteCarlo_organ_dose(self.mc.old_MUs, 'CG_MC')
        #  mc095_plan        = self.load_MonteCarlo_organ_dose(self.mc.old_MUs, 'CG_monteCarlo2')
        CG_MC_MURefined_plan = self.load_MonteCarlo_organ_dose(unpickle_object(os.path.join(self.hparam.refined_segments_MUs_file,'optimized_MUs.pickle')), 'CG_MC_MURefined')
        CG_MC_JY_plan        = self.load_JYMonteCarlo_organ_dose('CG_JY_MC', '/mnt/win_share2/20200918_NPC_MCDOse_verify_by_JY_congliuReCal/dpm_result*Ave.dat')
        #  JYMC095_plan      = self.load_JYMonteCarlo_organ_dose('jymc0.95', '/mnt/win_share2/20200918_NPC_MCDOse_verify_by_JY_congliuReCal/dpm_result*Ave.dat', 0.95)
        #  CLRecalJYMC_plan  = self.load_JYMonteCarlo_organ_dose('jymc_clRecalc', '/mnt/win_share2/20200918_NPC_MCDOse_verify_by_JY/dpm_result*Ave.dat')

        #  self.comparison_plots([mc095_plan, mc_plan])
        #  self.comparison_plots([JYMC_plan, CLRecalJYMC_plan])
        #  self.comparison_plots([mc_plan, JYMC_plan])
        #  self.comparison_plots([CG_Depos_plan, CG_MC_plan, CG_MC_JY_plan, CG_MC_MURefined_plan])
        self.comparison_plots([CG_MC_plan, CG_MC_JY_plan, CG_MC_MURefined_plan])
        #  self.comparison_plots([JYMC095_plan, JYMC_plan])
        #  self.comparison_plots([mc_plan, cg_plan])
        #  self.comparison_plots([mc_plan, rf_plan])
        #  self.comparison_plots([cg_plan, rf_plan])
        #  self.comparison_plots([cg_plan, jymc_plan])

def eval():
    e = Evaluation(hparam)
    e.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    # run mode
    parser.add_argument("--eval", action='store_true',  help="eval")
    parser.add_argument("--test", action='store_true',  help="test")
    parser.add_argument("--optim", action='store_true', help="optimization")
   
    # exp_name
    parser.add_argument('--exp_name', default='0922_MU_MonteCarlo_Refine', type=str, help='experiment name', required=True)
    
    # Monte Carlo parameters 
    parser.add_argument("--Calculate_MC_unit_doses", action='store_true', help='if true, running FM.exe and gDPM.exe on winServer ')
    parser.add_argument('--optimized_segments_MUs_file', default='./results/0902_aperture_refine/', type=str, help='we will optimize MUs saved in this directory')
    parser.add_argument('--patient_ID', default='Pa14Plan53Rx53GPU_2', type=str, help='gDPM.exe will read and save data into this ID directory')
    parser.add_argument('--MCDose_shape', default=[167,256,256], type=list, help='to be consistent with gDPM.exe which uses this shape to process dicom.')
    parser.add_argument('--winServer_nb_threads', default=5, type=int, help='number of threads used for gDPM.exe. Too many threads may overflow the GPU memory')
    
    # TPS parameters: depostion matrix, etc.
    parser.add_argument('--dose_scale', default=1/1000, type=float, help='')
    parser.add_argument('--priority_scale', default=1/100, type=float, help='')
    parser.add_argument('--max_inten', default=5, type=int, help='max intensity when plot fluence map')
    parser.add_argument('--deposition_file', default='./data/Deposition_Index.txt', type=str, help='deposition matrix from tps')
    parser.add_argument('--deposition_pickle_file', default='/mnt/ssd/tps_optimization/NPC/deposition.pickle', type=str, help='parsed deposition matrix will be stored in this file')
    parser.add_argument('--valid_ray_file', default='data/ValidMatrix.txt', type=str, help='tps uses this file to indicate the correspondence between the bixels of fluence map and the rays of depostion matrix.')
    parser.add_argument('--cvs_table_file', default='data/27_OrganInfo.csv', type=str, help='this file defines optimization objectives')
    parser.add_argument("--is_check_ray_idx_order", action='store_true', help='')

    # optimization parameters
    parser.add_argument('--logs_interval', default=15, type=int, help='log optimization info at this rate')
    parser.add_argument('--steps', default=6000, type=int, help='total iter to run optimization')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('--plateau_patience', default=500, type=int, help='early stop the optimization when loss does not decrease this times')
    parser.add_argument('--optimizer_name', default='adam', type=str, help='optimizer')
    parser.add_argument('--scheduler_name', default='CosineAnnealingLR', type=str, help='learning rate sheduler')
    parser.add_argument('--device', default='cpu', type=str, help='cpu | cuda; PLS use cpu, because we can not afford the memory usage of MU*MCDose')
    
    # misc parameters
    hparam = parser.parse_args()
    parser.add_argument('--unitMUDose_npz_file', default='./data/MonteCarlo/'+hparam.patient_ID+'/unitMUDose.npz', type=str, help='MCDose fetched from winServer will be save in this file')
    parser.add_argument('--winServer_MonteCarloDir', default='/mnt/win_share/'+hparam.patient_ID, type=str, help='gDPM.exe save MCDose into this directory; this directory is shared by winServer')
    parser.add_argument('--refined_segments_MUs_file', default='./results/'+hparam.exp_name, type=str, help='optimized MUs will be save in this directory')
    parser.add_argument('--tensorboard_log', default='./logs/'+hparam.exp_name, type=str, help='tensorboard directory')
    hparam = parser.parse_args()

    if args.optim:
        optim(hparam)
    elif args.eval:
        eval()
    elif args.test:
        test(hparam)
    else:
        raise NotImplementedError
