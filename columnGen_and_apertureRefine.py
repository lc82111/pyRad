#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage.measurements import label 
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter

import os
import pdb
from argparse import ArgumentParser
from termcolor import colored, cprint
from io import StringIO
import sys
import collections
import shutil
import pandas as pd
import pickle

from utils import *
from data import Data
from loss import Loss
#  from sp import sp_solvers
from options import BaseOptions


def _smallest_contiguous_sum_2d(grad_map):
    ''' Arguments: 
         grad_map: 2d matrix
        Return: 
         segment: 1D segment vector; ndarray flatten bool vector with 1 indicates the opened bixel. 
         lrs: left and right leaf positions; ndarray (H, 2); e.g. for segment 011100, l=1, r=4 '''
    def _smallest_contiguous_sum(array):
        ''' find smallest contiguous sum for a row of gradMap
             Arguments: array: 1D vector
             Return:    1D vector '''
        cumulative_value, global_max, reduced_cost = 0, 0, 0
        c1, c1s = -1, -1 
        c2, c2s =  0, 0 
        while c2 < len(array):
            cumulative_value += array[c2]
            if cumulative_value >= global_max:
                global_max = cumulative_value 
                c1 = c2
            if (cumulative_value - global_max) < reduced_cost:
                reduced_cost = cumulative_value - global_max
                c1s = c1
                c2s = c2+1
            c2 = c2+1
        segment = np.zeros_like(array, dtype=np.bool)
        segment[c1s+1:c2s] = True
        return segment, [c1s+1, c2s]

    row_segs, lrs = [], []
    for row in grad_map:
        seg, lr = _smallest_contiguous_sum(row)
        row_segs.append(seg)
        lrs.append(lr)
    segment  = np.concatenate(row_segs, axis=0) # a long flatten bool vector
    lrs = np.asarray(lrs)  # (H, 2)
    return segment, lrs

class SubProblem():
    def __init__(self, hparam, loss, data):
        self.hparam, self.loss, self.data = hparam, loss, data
    
    def solve(self, dict_gradMaps):
        ''' Find aperture shapes with the minimum gradient based on gradMaps.
           Arguments:
                 dict_gradMaps: {beam_id: gradient maps; ndarray; matrix}
           Return: 
                 dict_segments: {beam_id: aperture shapes; bool ndarray; vector}
           '''
        cprint('solving SubProblem .............', 'yellow')
        dict_segments, dict_lrs = OrderedBunch(), OrderedBunch()
        for beam_id, grad_map in dict_gradMaps.items():
            blocked_bixels = ~self.data.dict_rayBoolMat[beam_id] # where 1 indicates non-valid/blocked bixel 
            grad_map[blocked_bixels] = 10000. # set nonvalid bixels as 10000 to enforce smallest_contiguous_sum() choose the smaller region 
            grad_map = grad_map.astype(np.float64)
            dict_segments[beam_id], dict_lrs[beam_id] = _smallest_contiguous_sum_2d(grad_map)
        cprint('done', 'green')
        return dict_segments, dict_lrs

class MasterProblem():
    def __init__(self, hparam, loss, data, sp):
        self.hparam, self.data, self.loss, self.sp =  \
             hparam,      data,      loss,      sp

        self.optimizer = Optimization(hparam, loss, data)
        
        # recode all segments and left/right leaf positions 
        self.dict_segments = OrderedBunch()
        self.dict_lrs      = OrderedBunch()

    def init_segments(self):
        '''return: 
        dict_gradMaps {beam_id: matrix}
        new_dict_segments {beam_id: vector}'''
        # deposition matrix (#voxels, #bixels)
        deposition = convert_depoMatrix_to_tensor(self.data.deposition, self.hparam.device)

        # get fluence 
        if self.hparam.optimization_continue:  # continue last optimization
            file_name = os.path.join(self.hparam.optimized_segments_MUs_file_path, 'optimized_segments_MUs.pickle')
            if not os.path.isfile(file_name):
                raise ValueError(f'file not exist: {file_name}')
            else:
                cprint(f'continue last optimization from {file_name}', 'yellow')
                segments_and_MUs = unpickle_object(file_name)
                dict_segments, dict_MUs = OrderedBunch(), OrderedBunch()
                for beam_id, seg_MU in segments_and_MUs.items():
                    dict_segments[beam_id] = torch.tensor(seg_MU['Seg'], dtype=torch.float32, device=self.hparam.device)
                    dict_MUs[beam_id]      = torch.tensor(seg_MU['MU'],  dtype=torch.float32, device=self.hparam.device, requires_grad=True)
                fluence, _ = computer_fluence(self.data, dict_segments, dict_MUs)
                self.dict_segments = OrderedBunch()
                for beam_id, seg in dict_segments.items(): self.dict_segments[beam_id] = seg.cpu().numpy()
        else:
            fluence = torch.zeros((deposition.size(1),), dtype=torch.float32, device=self.hparam.device, requires_grad=True)  # (#bixels,)

        # compute fluence gradient
        doses = cal_dose(deposition, fluence) # cal dose (#voxels, )
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)  # split organ_doses to obtain individual organ doses
        loss, breaking_points_nums = self.loss.loss_func(dict_organ_doses)
        print(f'breaking points #: ', end='')
        for organ_name, breaking_points_num in breaking_points_nums.items(): print(f'{organ_name}: {breaking_points_num}   ', end='')
        print(f'loss={to_np(loss)}\n\n')
        loss.backward(retain_graph=False) # backward to get grad
        grads = fluence.grad.detach().cpu().numpy() # (#bixels,)

        # project 1D grad vector (#vaild_bixels,) to 2D fluence maps {beam_id: matrix}
        dict_gradMaps = self.data.project_to_fluenceMaps(grads)  # {beam_id: matrix}

        new_dict_segments, new_dict_lrs = self.sp.solve(dict_gradMaps) # new_dict_segments {beam_id: vector}

        del fluence, doses, deposition
        return dict_gradMaps, new_dict_segments, new_dict_lrs

    def update_segments_lrs(self, new_dict_segments, new_dict_lrs):
        ''' append the new segment vector to the column of old seg matrix
        Arguments:new_dict_segments {beam_id: ndarray (HxW,)}
                  new_dict_lrs      {beam_id: ndarray (H, 2)}
        Return: 
                self.dict_segments {beam_id: ndarray (HxW, #aperture)}
                self.dict_lrs      {beam_id: ndarray (#aperture, H, 2)}
        '''
        if len(self.dict_segments) == 0: # first new segment from sp.solve()
            for beam_id, new_seg in new_dict_segments.items():
                self.dict_segments[beam_id] = new_seg.reshape(-1,1)  # vector to matrix (#bixels,1)
                self.dict_lrs[beam_id] = np.expand_dims(new_dict_lrs[beam_id], 0)  # (1, H, 2)
        else: # continue optimization
            for beam_id, new_seg_vec in new_dict_segments.items():
                # append the new segment vector to the column of the old seg matrix 
                old_seg_mat = self.dict_segments[beam_id]
                self.dict_segments.update({beam_id: np.column_stack([old_seg_mat, new_seg_vec])})
                
                # append the new lrs to the first axis of the old lrs
                if 1 not in self.dict_lrs:  # continue optim from disk file
                    H, W = self.data.dict_rayBoolMat[beam_id].shape
                    old_lrs = restore_lrs(old_seg_mat, H, W)
                else:  # norm continue optim 
                    old_lrs = self.dict_lrs[beam_id]  # (#aperture, H, 2)
                new_lrs = np.expand_dims(new_dict_lrs[beam_id], 0)  # (1, H, 2)
                self.dict_lrs.update({beam_id: np.concatenate([old_lrs, new_lrs], 0)})  # (#aperture, H, 2)

    def solve(self, new_dict_segments, new_dict_lrs, nb_apertures):
        ''' adjust MUs and segs (lrs and partialExp), return the gradMaps for modified MUs and segs
        Arguments: new_dict_segments: {beam_id: ndarray vector}
        Return: dict_gradMaps: {beam_id: ndarray matrix}
        '''
        cprint('solving MasterProblem .............', 'yellow')
        self.update_segments_lrs(new_dict_segments, new_dict_lrs)

        # optim
        grad, self.dict_MUs, self.dict_segments, self.dict_lrs, self.dict_partialExp, self.loss = self.optimizer.run(
                                                            self.dict_segments, self.dict_lrs,
                                                            nb_apertures,
                                                            self.hparam.learning_rate,
                                                            self.hparam.master_steps,
                                                            self.hparam.optimizer_name,
                                                            self.hparam.scheduler_name) 

        # project 1D grad vector (#vaild_bixels,) to 2D fluence maps {beam_id: matrix}
        dict_gradMaps = self.data.project_to_fluenceMaps(grad)  # {beam_id: matrix}
        cprint('done', 'green')

        return dict_gradMaps 

class Optimization():
    def __init__(self, hparam, loss, data):
        '''
        using torch to optimize the master problem.
        '''
        self.hparam, self.loss, self.data = hparam, loss, data
        self.global_step = 0

        self.deposition = convert_depoMatrix_to_tensor(self.data.deposition, hparam.device)

        # create a tensorboard summary writer using the specified folder name.
        if hparam.logs_interval != None:
            self.tb_writer = SummaryWriter(hparam.tensorboard_log)
            tmp_hparam = (vars(self.hparam)).copy()  #  access the namespace's dictionary with vars()
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

    def _modulate_segment_with_partialExposure(self, segs, pes, lrs, rayMat):
        '''
        Arguments:
            seg: tensor (HxW, #aperture)
            pe: tensor (#aperture, H, 2)
            lrs: ndarray (#aperture, H, 2)
        Return: 
            modulated_segs (HxW, #aperture)
        '''
        H, W = rayMat.shape

        pe_3d = []
        for i, alrs in enumerate(lrs):  # for each aperture; 
            pe_2d = []
            for j, lr in enumerate(alrs): # for each row
                l, r = lr
                assert (l<r-1 and l>=0 and r<=W)
                # NOTE: torch.ones((0,)) works and return a empty tensor
                before = torch.ones((l,),     dtype=torch.float32, device=self.hparam.device)
                middle = torch.ones((r-l-2,), dtype=torch.float32, device=self.hparam.device)
                after  = torch.ones((W-r,),   dtype=torch.float32, device=self.hparam.device)
                vector = torch.cat([before, pes[i,j,0].view(1), middle, pes[i,j,1].view(1), after], dim=0)  # (W, ) , set up this vector is the only way to backward grad to pes 
                pe_2d.append(vector)
            pe_3d.append(torch.cat(pe_2d, dim=0))  # (HxW,)
        pe_3d = torch.stack(pe_3d, dim=1) # (HxW, #aperture)
        modulated_segs = pe_3d * segs

        return modulated_segs 

    def computer_fluence(self, dict_segments, dict_partialExp, dict_lrs, dict_MUs):
        '''computer fluence from seg and MU. 
        Arguments:
            dict_segments: {beam_id: tensor consists of segment columns (hxw, #aperture)}
            dict_partialexp: {beam_id: tensor (#aperture, h, 2)}
            dict_lrs: {beam_id: ndarray (#aperture, h, 2)}
            dict_mus:{beam_id: tensor (#aperture,)}
        Return: 
            fluence (#valid_bixels, )
        For retain_grad() see : https://discuss.pytorch.org/t/how-do-i-calculate-the-gradients-of-a-non-leaf-variable-w-r-t-to-a-loss-function/5112
        '''
        dict_fluences = OrderedBunch()
        for beam_id in range(1, len(dict_segments)+1):
            # 0. modulate segment with partial exposure
            pe = torch.sigmoid(dict_partialExp[beam_id])  # [0,1] constraint
            seg = self._modulate_segment_with_partialExposure(dict_segments[beam_id], pe, dict_lrs[beam_id], self.data.dict_rayBoolMat[beam_id])

            # 1. modulated_segment * MU
            MU = torch.abs(dict_MUs[beam_id]) # nonnegative constraint 
            dict_fluences[beam_id] = torch.matmul(seg, MU)  # {beam_id: vector}

        fluence, dict_fluenceMaps = self.data.project_to_validRays_torch(dict_fluences)  # (#valid_bixels,), {beam_id: matrix}
        fluence.retain_grad()
        return fluence, dict_fluenceMaps

    def _step_lrs_segments(self, dict_partialExp, dict_lrs, dict_segments):
        '''
        ajust leaf postions according optimized partialExp
        dict_partialExp: {beam_id: tensor (#aperture, H, 2)}
        dict_lrs:        {beam_id: ndarray (#aperture, H, 2)}
        dict_segments:   {beam_id: tensor (HxW, #aperture)}
        return:
            modified dict_lrs, dict_segments
        '''
        with torch.no_grad():  # see: https://discuss.pytorch.org/t/layer-weight-vs-weight-data/24271
            for i, lrs in dict_lrs.items(): # each beam
                H, W = self.data.dict_rayBoolMat[i].shape
                for j, alrs in enumerate(lrs): # each aperture
                    for k, lr in enumerate(alrs): # each row 
                        l,r = lr; r -= 1  # NOTE: r represents opened rightmost bixel 
                        l_pe, r_pe = to_np(torch.sigmoid(dict_partialExp[i].detach()))[j,k]

                        # checking the left leaf
                        cur_grad = to_np(dict_segments[i].grad)[:,j][k*W+l]
                        if l_pe>0.95 and l-1>0:  # open left closed bixel / leaf move to left?
                            left_grad = to_np(dict_segments[i].grad)[:,j][k*W+(l-1)]
                            if left_grad<0: # left grad<0 ==> increase left bixel intensity (current left intensity=0) to decrease the loss 
                                if cur_grad<0 or abs(left_grad) > abs(cur_grad): # cur grad<0 ==> increase current bixel (opened already);
                                    dict_lrs[i][j,k,0] -= 1 # move to left
                                    dict_segments[i][:,j][k*W:k*W+W][l-1] = 1
                                    dict_partialExp[i][j,k,0] = 0.
                        elif l_pe<0.05 and l+1<W-2 and l+1<r: # close current bixel / move to right?  
                            if cur_grad>0: # cur grad>0 ==> decrease current bixel (opened already);
                                dict_lrs[i][j,k,0] += 1 # move to right 
                                dict_segments[i][:,j][k*W:k*W+W][l] = 0
                                dict_partialExp[i][j,k,0] = 0.

                        # checking the right leaf
                        cur_grad = to_np(dict_segments[i].grad)[:,j][k*W+r]
                        if r_pe>0.95 and r+1<W: # open right closed bixel / leaf move to right?
                            right_grad = to_np(dict_segments[i].grad)[:,j][k*W+(r+1)]
                            if right_grad<0: 
                                if cur_grad<0 or abs(cur_grad)<abs(right_grad):
                                    dict_lrs[i][j,k,1] += 1 # move to right 
                                    dict_segments[i][:,j][k*W:k*W+W][r+1] = 1
                                    dict_partialExp[i][j,k,1] = 0.
                        elif r_pe<0.05 and r-1>1 and l<r-1: # close current bixel / leaf move to left?
                            if cur_grad>0: # cur grad>0 ==> decrease current bixel (opened already);
                                dict_lrs[i][j,k,1] -= 1 # move to left 
                                dict_segments[i][:,j][k*W:k*W+W][r] = 0
                                dict_partialExp[i][j,k,1] = 0.

                        # ensuring distinct l r 
                        l,r = dict_lrs[i][j,k]; r -= 1 # new l r
                        if not l < r:
                            left_grad  = to_np(dict_segments[i].grad)[:,j][k*W+(l-1)]
                            right_grad = to_np(dict_segments[i].grad)[:,j][k*W+(l+1)]
                            # left leaf move to left
                            if left_grad < right_grad or r+1>=W:
                                if l-1>0:
                                    dict_lrs[i][j,k,0] -= 1 # move to left
                                    dict_segments[i][:,j][k*W:k*W+W][l-1] = 1
                                    dict_partialExp[i][j,k,0] = 0.
                            # right leaf move to right 
                            if left_grad > right_grad or l-1<0:
                                if r+1<W:
                                    dict_lrs[i][j][k][1] += 1 # move to right 
                                    dict_segments[i][:,j][k*W:k*W+W][r+1] = 1
                                    dict_partialExp[i][j,k,1] = 0.

                            # double check
                            l,r = dict_lrs[i][j,k]; r -= 1 # new new l r
                            if l>=r: # debug
                                pdb.set_trace()
                                print(l,r)
                            assert l != r

    def _get_partial_exposure_tensor(self, lrs, seg, rayMat):
        '''
        Set partial exposed (pe) variable tensors for optimization, and modify the segment elements corresponding the pes. 
        lrs: ndarray (#aperture, H, 2), 2=(l,r)
        seg: ndarray (HxW, #aperture)

        NOTE: 
            left/right leaf postion (l,r) will make bixels [ABC] expose.
            [l=0] A B C [r=3] D

        return:
            pe: tensor (#aperture, H, 2)
        '''
        H, W = rayMat.shape
        def get_distinct_lr(l, r):  # let l!=r and l!=r-1, ensure l<=r-2
            if l==r:  # row closed
                if   l>0:  l -= 1
                elif r<W:  r += 1
                else: raise NotImplementedError
            if l==r-1:  # only one bixel exposed
                if   l>0:  l -= 1
                elif r<W:  r += 1
                else: raise NotImplementedError
            return [l, r]
        
        # set partial exposure tensor
        pes = torch.full(lrs.shape, fill_value=0., dtype=torch.float32, device=self.hparam.device, requires_grad=True) # sigmoid(0)=0.5

        # set lrs
        for i, aperture in enumerate(lrs):  # for each aperture
            for j, lr in enumerate(aperture):  # for each row
                [l, r] = get_distinct_lr(lr[0], lr[1])
                lrs[i,j] = [l,r]
                seg[j*W:j*W+W, i] [[l,r-1]] = 1

        return pes, seg

    def run(self, dict_segments, dict_lrs, nb_apertures, learning_rate, steps, optimizer_name, scheduler_name):
        ''' adjust the MU and segments (lrs), via optimizing MU and partial_exp
        Arguments:
             dict_segments: {beam_id: ndarray (HxW, #aperture)}
             dict_lrs:      {beam_id: ndarray (#aperture, H, 2)}
           Return:
             optimized/modified dict_segments: {beam_id: ndarray (HxW, #aperture)}
             optimized/modified dict_lrs:      {beam_id: ndarray (#aperture, H, 2)}
             optimized/modified dict_partialExp:{beam_id: ndarray (#aperture, H, 2)}
             optimized/modified dict_MUs:      {beam_id: ndarray (#aperture,)}
        '''
        # set up new tensors to be optimized
        self.nb_apertures = nb_apertures
        dict_MUs, dict_partialExp = OrderedBunch(), OrderedBunch()
        for beam_id, seg in dict_segments.items():
            dict_MUs[beam_id] = torch.rand((seg.shape[1],), dtype=torch.float32, device=self.hparam.device, requires_grad=True) # random [0, 1]
            dict_partialExp[beam_id], seg = self._get_partial_exposure_tensor(dict_lrs[beam_id], seg, self.data.dict_rayBoolMat[beam_id])
            dict_segments.update({beam_id: torch.tensor(seg, dtype=torch.float32, device=self.hparam.device, requires_grad=True)})

        # optimizer
        optimizer, scheduler = self._get_optimizer(list(dict_MUs.values()) + list(dict_partialExp.values()), learning_rate, steps, optimizer_name, scheduler_name)

        # loop
        min_loss, patience = np.inf, 0
        for i in range(steps):
            # forward
            self.global_step += 1
            fluence, loss = self.forward(self.deposition, dict_segments, dict_partialExp, dict_lrs, dict_MUs) # (#vaild_bixels,)

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=False) # acccumulate gradients

            # best state
            if to_np(loss) < min_loss:
                min_loss = to_np(loss)
                patience = 0
                self._save_best_state(min_loss, fluence.grad.detach().cpu().numpy(), dict_MUs, dict_segments, dict_lrs, dict_partialExp)

            # optim
            optimizer.step() # do gradient decent w.r.t MU and partialExp
            self._step_lrs_segments(dict_partialExp, dict_lrs, dict_segments) # ajust leafs position in place: dict_segments and dict_lrs
            scheduler.step() # adjust learning rate

            # early stop
            if to_np(loss) > min_loss:
                patience += 1
            if patience > self.hparam.plateau_patience:
                cprint(f'Loss dose not drop in last {patience} iters. Early stopped.', 'yellow')
                break

        cprint(f'optimization done.\n Min_loss={min_loss}', 'green')

        return self.best_grads, self.best_dict_MUs, self.best_dict_segments, self.best_dict_lrs, self.best_dict_partialExp, self.best_loss
    
    def _save_best_state(self, loss, grad, dict_MUs, dict_segments, dict_lrs, dict_partialExp):
        self.best_loss = loss

        # grad of fluence
        self.best_grads = grad # (#vaild_bixels,)

        self.best_dict_MUs, self.best_dict_segments, self.best_dict_lrs, self.best_dict_partialExp = OrderedBunch(), OrderedBunch(), OrderedBunch(), OrderedBunch()
        for beam_id, lrs in dict_lrs.items():
            self.best_dict_MUs[beam_id]        = to_np(dict_MUs[beam_id])
            self.best_dict_segments[beam_id]   = to_np(dict_segments[beam_id])
            self.best_dict_partialExp[beam_id] = to_np(dict_partialExp[beam_id])
            self.best_dict_lrs[beam_id]        = lrs

    def forward(self, deposition, dict_segments, dict_partialExp, dict_lrs, dict_MUs):
        '''
        0. compute fluence from seg, mu, and pe
        1. compute dose from deposition and fluence.
        2. compute loss from dose

        dict_segments: {beam_id: matrix consists of segment columns}
        dict_partialExp: {beam_id: (#aperture, H, 2)}
        dict_lrs: {beam_id: (#aperture, H, 2)}
        dict_MUs:{beam_id: vector of segment MU}
        '''
        fluence, self.dict_fluenceMaps = self.computer_fluence(dict_segments, dict_partialExp, dict_lrs, dict_MUs) # (#valid_bixels,), {beam_id: matrix}
        doses = cal_dose(deposition, fluence)
        loss = self.loss_func(doses, dict_segments, dict_MUs) # (1,)
        return fluence, loss

    def loss_func(self, doses, dict_segments, dict_MUs):
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)
        
        # cal loss
        loss, breaking_points_nums = self.loss.loss_func(dict_organ_doses)

        # tensorboard logs 
        self.tb_writer.add_scalar('nb_apertures', self.nb_apertures, self.global_step)
        self.tb_writer.add_scalar('loss/total_loss', loss, self.global_step)
        self.tb_writer.add_scalars('BreakPoints', breaking_points_nums, self.global_step)
        if self.global_step % self.hparam.logs_interval == 0:
            print(f"\n nb_apertures={self.nb_apertures}. solving master problem: global_step={self.global_step}------------------------------------------  ")
            # print breaking_points_nums
            print('breaking points #: ', end='')
            for organ_name, breaking_points_num in breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            # MU hist
            for beam_id, MU in dict_MUs.items():
                self.tb_writer.add_histogram(f'MUs_hist/{beam_id}', torch.abs(MU), self.global_step)  # nonnegative constraint
            # dose histogram
            for organ_name, dose in dict_organ_doses.items():
                if dose.size(0) != 0:
                    self.tb_writer.add_histogram(f'dose histogram/{organ_name}', dose, self.global_step)
            # fluence and segment maps 
            for beam_id, FM in self.dict_fluenceMaps.items():
                self.tb_writer.add_image(f'FluenceMaps/{beam_id}', FM, self.global_step, dataformats='HW')
                high, width = FM.size()
                segs = dict_segments[beam_id].detach() # matrix
                for col_idx in range(segs.size(1)):
                    seg = segs[:,col_idx].view(high, width)
                    self.tb_writer.add_image(f'BeamSeg{beam_id}/{col_idx}', seg.to(torch.uint8)*255, self.global_step, dataformats='HW')

            print("\n loss={:.6f} \n".format(to_np(loss)))

        return loss

def save_result(mp): 
    def _modulate_segment_with_partialExposure(seg, lrs, pes):
        '''
        Imposing the partialExp effect at the endpoint of leaf
        lrs: (#aperture, H, 2); seg:(HxW, #aperture); pes:(#aperture, H, 2)
        '''
        for i, aperture in enumerate(lrs):  # for each aperture
            for j, lr in enumerate(aperture):  # for each row
                assert label(seg[j*W:j*W+W, i])[1] <=1  # ensure only zero or one connected component in a row
                [l, r] = lr
                l_pe, r_pe = sigmoid(pes[i, j])
                # close hopeless bixel?
                if l_pe < 0.6:
                    seg[j*W:j*W+W, i] [l] = 0
                if r_pe < 0.6:
                    seg[j*W:j*W+W, i] [r-1] = 0
        return seg

    results = OrderedBunch()
    for (beam_id, MU), (_, seg) in zip(mp.dict_MUs.items(), mp.dict_segments.items()):
        H, W = mp.data.dict_rayBoolMat[beam_id].shape
        
        validRay = mp.data.dict_rayBoolMat[beam_id].flatten().reshape((-1,1)) # where 1 indicates non-valid/blocked bixel 
        validRay = np.tile(validRay, (1, seg.shape[1]))  # (HxW, #aperture)
        seg = seg*validRay  # partialExp may open bixels in non-valid regions.

        lrs = mp.dict_lrs[beam_id]  # (#aperture, H, 2)
        pes = mp.dict_partialExp[beam_id] # (#aperture, H, 2)
        seg = _modulate_segment_with_partialExposure(seg, lrs, pes)

        results[beam_id] = {'MU': np.abs(MU), 'Seg': seg, 'lrs':lrs, 'PEs':pes, 'global_step':mp.optimizer.global_step} 

    if not os.path.isdir(hparam.optimized_segments_MUs_file_path):
        os.makedirs(hparam.optimized_segments_MUs_file_path)
    pickle_object(os.path.join(hparam.optimized_segments_MUs_file_path,'optimized_segments_MUs.pickle'), results)

def main(hparam):
    if not hparam.optimization_continue:
        del_fold(hparam.tensorboard_log)  # clear log dir, avoid the messing of log dir 

    # init data and loss
    data = Data(hparam)
    loss = Loss(hparam, data.csv_table)

    # init sub- and master- problem
    sp = SubProblem(hparam, loss, data)
    mp = MasterProblem(hparam, loss, data, sp)

    # master and sp loop 
    nb_apertures = 0
    dict_gradMaps, next_dict_segments, next_dict_lrs = mp.init_segments()
    while multiply_dict(dict_gradMaps, next_dict_segments)<0 and nb_apertures<hparam.nb_apertures:  # next_seg * cur_grad < 0 means open next_seg (intensity of bixels - negative grad == increase the intensity) will decrease the loss 
        dict_gradMaps = mp.solve(next_dict_segments, next_dict_lrs, nb_apertures)  #  {beam_id: matrix}
        next_dict_segments, next_dict_lrs = sp.solve(dict_gradMaps)     # {beam_id: bool vector}
        nb_apertures += 1
        cprint(f'nb_apertures: {nb_apertures} done.', 'green')

    # save optimized segments and MUs
    pdb.set_trace()
    save_result(mp)

    # release memory
    torch.cuda.empty_cache()

    cprint('all done!!!', 'green')


if __name__ == "__main__":
    hparam = BaseOptions().parse()
    main(hparam)
