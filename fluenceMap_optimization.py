#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from bunch import Bunch
from scipy.ndimage import median_filter 
import numpy.random as npr
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter

import os
from termcolor import colored, cprint
from io import StringIO
import sys
import collections
import shutil
import pdb
import pandas as pd
from tqdm.auto import tqdm
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc
from utils import *
from data import Data
from loss import Loss
from options import BaseOptions


def constraint_intensity(ray_inten, max_inten):
    # intensity constraints: 0 <= ray_inten <= max_inten   
    ray_inten = torch.abs(ray_inten)
    ray_inten = torch.where(ray_inten>max_inten, max_inten, ray_inten)  # ray_inten <= max_inten
    return ray_inten

class Optimization():
    def __init__(self, hparam, loss, data):
        self.hparam = hparam
        self.loss = loss
        self.data = data

        # (#voxels, #bixels)
        self.deposition = torch.tensor(data.deposition, dtype=torch.float32, device=self.hparam.device)
        # max intensity
        self.max_fluence = torch.full((self.deposition.size(1),), self.hparam.max_fluence, dtype=torch.float32, device=self.hparam.device)

        # create a tensorboard summary writer using the specified folder name.
        self.tb_writer = SummaryWriter(hparam.tensorboard_log)

        #torch.random.manual_seed(0)
        #np.set_printoptions(precision=4, sign=' ')

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

    def run(self, learning_rate, steps, optimizer_name, scheduler_name):
        # var to be optimized
        fluence = torch.rand((self.deposition.size(1),), dtype=torch.float32, device=self.hparam.device, requires_grad=True) # [0, 1]

        # optimizer
        optimizer, scheduler = self._get_optimizer([fluence,], learning_rate, steps, optimizer_name, scheduler_name)
        
        # optim loop
        self.step, min_loss, patience = 0, np.inf, 0
        for i in range(steps):
            # forward
            self.step += 1
            loss = self.forward(self.deposition, fluence)

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=False) # acccumulate gradients

            # best state
            if to_np(loss) < min_loss:
                min_loss = to_np(loss)
                patience = 0
                best_fluence = fluence.detach().cpu() # (#vaild_bixels,)

            # optim
            optimizer.step() # do gradient decent
            scheduler.step() # adjust learning rate

            # early stop
            if to_np(loss) > min_loss:
                patience += 1
            if patience > self.hparam.plateau_patience:
                cprint(f'Loss dose not drop in last {patience} iters. Early stopped.', 'yellow')
                break

        cprint(f'optimization done.\n Min_loss={min_loss}', 'green')
        return best_fluence 

    def forward(self, deposition, fluence):
        fluence = constraint_intensity(fluence, self.max_fluence)
        doses = torch.matmul(deposition, fluence) # a matrix multiply a vector
        loss = self.loss_func(doses, fluence)
        return loss

    def loss_func(self, doses, fluence):
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organ_inf)
        
        # cal loss
        loss, breaking_points_nums, smooth = self.loss.loss_func(dict_organ_doses, fluence, self.data)
        
        # logs
        # tensorboard log
        self.tb_writer.add_scalar('loss/loss', loss, self.step)
        self.tb_writer.add_scalar('loss/smooth_regularizer', smooth, self.step)
        self.tb_writer.add_scalars('BreakPoints', breaking_points_nums, self.step)

        if self.step % self.hparam.logs_interval == 0:
            print(f"\n total iter={self.step}------------------------------------------  ")

            # print breaking_points_nums
            print('breaking points #: ', end='')
            for organ_name, breaking_points_num in breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            print("\n loss={:.6f}; smooth={:.6f} \n".format(to_np(loss), to_np(smooth)))

            # fluence hist
            self.tb_writer.add_histogram('fluence histogram', fluence, self.step)

            # dose histogram
            for organ_name, dose in dict_organ_doses.items():
                if dose.size(0) != 0:
                    self.tb_writer.add_histogram(f'dose histogram/{organ_name}', dose, self.step)

            # fluence map 
            for beam_id, fm in self.loss.dict_FluenceMap.items():
                self.tb_writer.add_image(f'FluenceMaps/{beam_id}', fm, self.step, dataformats='HW')

        return loss + smooth


def main(hparam):
    if not hparam.optimization_continue:
        del_fold(hparam.tensorboard_log)  # clear log dir, avoid the messing of log dir 

    data = Data(hparam)
    loss = Loss(hparam, data.csv_table)
    
    # optim
    optim = Optimization(hparam, loss, data)
    fluence = optim.run(hparam.learning_rate, hparam.steps, hparam.optimizer_name, hparam.scheduler_name) 

    # save 
    fluence = constraint_intensity(fluence, optim.max_fluence)
    pickle_object(hparam.optimized_fluence_file_path+'/optimized_fluence.pickle', to_np(fluence))


if __name__ == "__main__":
    hparam = BaseOptions().parse()
    main(hparam)
