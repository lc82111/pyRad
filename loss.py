#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
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

class Loss():
    def __init__(self, hparam, allOrganTable, is_warning=True):
        self.hparam         = hparam
        self.priority_scale = hparam.priority_scale
        self.dose_scale     = hparam.dose_scale
        self.allOrganTable  = allOrganTable 
        self.is_warning     = is_warning 

    def _dose_window(self, dose, constraint, weight):
        min_dose = float(constraint['Min Dose'])*self.dose_scale
        max_dose = float(constraint['Max Dose'])*self.dose_scale

        error = torch.cat([dose[dose<min_dose]-min_dose, dose[dose>max_dose]-max_dose])
        #assert to_np(torch.isnan(error)).any() == False
        loss = (error**2).mean() * weight
        breakNum = int(to_np((dose<min_dose).sum()) + to_np((dose>max_dose).sum()))
        return loss, breakNum

    def _dose_mean(self, dose, constraint, weight):
        mean_dose = float(constraint['Min Dose'])*self.dose_scale

        error = dose.mean() - mean_dose
        loss = (error**2) * weight
        breakNum = int(to_np((dose!=mean_dose).sum()))
        return loss, breakNum

    def _dose_upper(self, dose, constraint, weight):
        max_dose = float(constraint['Max Dose'])*self.dose_scale

        error = dose[dose>max_dose] - max_dose
        # empty breaking points
        if len(error) == 0:
            error = torch.tensor(0.)

        loss = (error**2).mean() * weight
        breakNum = int(to_np((dose>max_dose).sum()))
        return loss, breakNum

    def _dvh_lower(self, dose, constraint, weight):
        min_dose = float(constraint['Min Dose'])*self.dose_scale
        dvh_vol  = float(constraint['DVH Volume'])

        k = int((1-dvh_vol) * len(dose))
        d, _ = torch.kthvalue(dose, k) # k th smallest element
        error = dose[(dose>=d) * (dose<=min_dose)] - min_dose
        # empty breaking points
        if len(error) == 0:
            error = torch.tensor(0.)

        loss = (error**2).mean() * weight
        breakNum = int(to_np(((dose>=d) * (dose<=min_dose)).sum()))
        return loss, breakNum

    def _dvh_upper(self, dose, constraint, weight):
        max_dose = float(constraint['Max Dose'])*self.dose_scale
        dvh_vol  = float(constraint['DVH Volume'])

        k = int((1-dvh_vol) * len(dose))
        d, _ = torch.kthvalue(dose, k) # k th smallest element
        error = dose[(dose>=max_dose) * (dose<=d)] - max_dose
        # empty breaking points
        if len(error) == 0:
            error = torch.tensor(0.)

        loss = (error**2).mean() * weight
        breakNum = int(to_np(((dose>=max_dose) * (dose<=d)).sum()))
        return loss, breakNum

    def loss_func(self, dict_organ_doses, fluence=None, data=None): 
        ''' Arguments: dict_organ_doses: {organ_name: dose tensor} '''
        total_loss = 0
        dict_breakNUM = collections.OrderedDict()

        for organ_name, constraint in self.allOrganTable.items():  # iter over columns
            # skip zero points organs
            if int(constraint['Points Number']) == 0:
               continue 

            # skip skin related organs
            if 'skin' in organ_name:
               if self.is_warning: cprint(f'Warning: skip skin related organ:{organ_name} in allOrganTable', 'red')
               continue 

            # get organ dose
            if organ_name in dict_organ_doses:
                dose = dict_organ_doses[organ_name] 
            elif organ_name.rsplit('.')[0] in dict_organ_doses: # conside duplicated organ_name
                dose = dict_organ_doses[organ_name.rsplit('.')[0]] 
            else:
                if self.is_warning: cprint(f'Warning: organ {organ_name} find in allOrganTable but not in dict_organ_doses', 'red')
                continue
                #raise ValueError
            
            # soft type
            if constraint['Hard/Soft'] =='SOFT':
                soft = 1/100.
            else:
                soft = 1
            
            # weight
            weight = float(constraint['Priority'])*self.priority_scale*soft

            # loss type
            vol_type, st_type = constraint['Volume Type'], constraint['Constraint Type'] 
            if  vol_type=='DOSE' and st_type=='WINDOW':
                loss, breakNum = self._dose_window(dose, constraint, weight)
            elif vol_type=='DOSE' and st_type=='UPPER':
                loss, breakNum = self._dose_upper(dose, constraint, weight)
            elif vol_type == 'DOSE' and st_type =='MEAN':
                loss, breakNum = self._dose_mean(dose, constraint, weight)
            elif vol_type == 'DVH' and st_type =='UPPER':
                loss, breakNum = self._dvh_upper(dose, constraint, weight)
            elif vol_type == 'DVH' and st_type =='LOWER':
                loss, breakNum = self._dvh_lower(dose, constraint, weight)
            else:
                raise NotImplemented
          
            debug = False
            if debug and loss.grad_fn == None and loss != 0:
                pdb.set_trace()
                pass

            # acumulate loss
            total_loss += loss
            dict_breakNUM[organ_name] = breakNum

        # smooth regularization for fluence map optim only!!! 
        if fluence is not None and data is not None:
            smooth = self._smooth_regularizer(fluence, data) * self.hparam.smooth_weight
            return total_loss, dict_breakNUM, smooth
        else:
            return total_loss, dict_breakNUM

    def _smooth_regularizer(self, fluence, data):
        def _spatial_gradient(F, mask):
            # F: fluence map contains ray intensities;
            # mask: 0/1 matrix with 1 indicate the present of ray.

            delta_r = F[1:,:] - F[:-1,:] # horizontal gradient (H-1, W) 
            delta_c = F[:,1:] - F[:,:-1] # vertical gradient   (H,   W-1)
            
            # using mask to cancel the unwant grad among the ray boundary. 
            mask_r = (mask[1:,:] - mask[:-1,:]).abs().logical_not().float()
            mask_c = (mask[:,1:] - mask[:,:-1]).abs().logical_not().float()
            delta_r = delta_r * mask_r
            delta_c = delta_c * mask_c

            delta_r = delta_r[1:,:-2]**2  # (H-2, W-2)
            delta_c = delta_c[:-2,1:]**2  # (H-2, W-2)
            delta   = torch.abs(delta_r + delta_c)

            epsilon  = 1e-8 # where is a parameter to avoid square root is zero in practice.
            rough = torch.mean(torch.sqrt(delta+epsilon)) # eq.(11) in the paper, mean is( used instead of sum.)
            return rough 
        
        loss = 0
        self.dict_FluenceMap = data.project_to_fluenceMaps_torch(fluence)
        for beam_id, fm in self.dict_FluenceMap.items():
            mask = data.dict_rayBoolMat[beam_id]
            mask = torch.tensor(mask, dtype=torch.float32, device=fm.device)
            loss += _spatial_gradient(fm, mask)
        return loss 
