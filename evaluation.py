#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr
import pandas as pd
from termcolor import cprint

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter

import os, pdb, sys, collections, pickle, shutil
from argparse import ArgumentParser, Namespace
from termcolor import colored, cprint
from io import StringIO


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc

from utils import *
from data import Data
from loss import Loss
from options import BaseOptions
from monteCarlo import MonteCarlo


class Evaluation():
    def __init__(self, hparam):
        self.hparam = hparam

        # init data and loss
        self.data = Data(hparam)
        self.loss = Loss(hparam, self.data.csv_loss_table)

        # deposition matrix (#doseGrid, #bixels)
        self.deposition = convert_depoMatrix_to_tensor(self.data.deposition, self.hparam.device)
        
        # MC dose
        if hparam.MCPlan or hparam.MCJYPlan or hparam.MCMURefinedPlan or hparam.NeuralDosePlan:
            self.mc = MonteCarlo(hparam, self.data)
            self.unitMUDose = self.mc.get_all_beams_unit_MCdose()  # unitMUDose, ndarray (nb_beams*nb_apertures, D, H, W) 

    def load_MonteCarlo_OrganDose(self, MUs, name, scale=1):
        MUs = np.abs(MUs) / self.hparam.dose_scale  # x1000
        MCdoses = self.unitMUDose * MUs * scale
        MCdoses = MCdoses.sum(axis=0, keepdims=False)  #  (D, H, W) 
        MCdoses = torch.tensor(MCdoses, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'dose':dict_organ_doses, 'name':name})

    def load_JYMonteCarlo_OrganDose(self, name, dosefilepath, scale=1):
        MCdoses = self.mc.get_JY_MCdose(dosefilepath) * scale
        MCdoses = torch.tensor(MCdoses, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'dose':dict_organ_doses, 'name':name})

    def load_Depos_OrganDose(self, name, scale=1):
        cprint(f'deposPlan uses following parameters:{self.hparam.optimized_segments_MUs_file_path}; {self.hparam.deposition_pickle_file_path}; {self.hparam.valid_ray_file};', 'yellow')

        # get seg and MU
        file_name = self.hparam.optimized_segments_MUs_file_path+'/optimized_segments_MUs.pickle'
        if not os.path.isfile(file_name): raise ValueError(f'file not exist: {file_name}')
        cprint(f'load segments and MUs from {file_name}', 'yellow')
        segments_and_MUs = unpickle_object(file_name)
        dict_segments, dict_MUs = OrderedBunch(), OrderedBunch()
        for beam_id, seg_MU in segments_and_MUs.items():
            dict_segments[beam_id] = torch.tensor(seg_MU['Seg'], dtype=torch.float32, device=self.hparam.device)
            dict_MUs[beam_id]      = torch.tensor(seg_MU['MU'],  dtype=torch.float32, device=self.hparam.device, requires_grad=True)

        # compute fluence
        fluence, _ = computer_fluence(self.data, dict_segments, dict_MUs)
        fluence    = fluence / self.hparam.dose_scale * scale # * 1000
        dict_FMs   = self.data.project_to_fluenceMaps(to_np(fluence))

        # compute dose
        doses = cal_dose(self.deposition, fluence)
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)
        
        return OrderedBunch({'fluence':to_np(fluence), 'dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def load_TPS_OrganDose(self, name='TPSOptimResult'):
        # intensity
        fluence = np.loadtxt(self.hparam.tps_ray_inten_file)
        fluence = torch.tensor(fluence, dtype=torch.float32, device=self.hparam.device)
        dict_FMs = self.data.project_to_fluenceMaps(to_np(fluence))

        # dose
        doses = cal_dose(self.deposition, fluence)
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)

        return OrderedBunch({'fluence':to_np(fluence), 'dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def load_fluenceOptim_OrganDose(self, name):
        # intensity
        fluence = loosen_object(os.path.join(self.hparam.optimized_fluence_file_path+'/optimized_fluence.pickle'))
        fluence = torch.tensor(fluence, device=self.hparam.device)

        fluence = torch.abs(fluence)
        fluence = torch.where(fluence>self.hparam.max_fluence, self.hparam.max_fluence, ray_inten)
        fluence = fluence / self.hparam.dose_scale # * 1000

        # compute dose
        doses = cal_dose(self.deposition, fluence)
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)
        
        return OrderedBunch({'fluence':to_np(fluence), 'dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def comparison_plots(self, plans):
        '''
        plans: list of plan
        '''
        ## print loss and breaking pts num
        for plan in plans: 
            dict_organ_doses = plan.dose.copy()
            for name, dose in dict_organ_doses.copy().items():
                dict_organ_doses.update({name: dose*self.hparam.dose_scale})  # /1000
            plan_loss, plan_breaking_points_nums = self.loss.loss_func(dict_organ_doses)
            print(f'{plan.name} breaking points #: ', end='')
            for organ_name, breaking_points_num in plan_breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            print(f'loss={plan_loss}\n\n')

        ## plot DVH
        # pop unnecessary organ dose to avoid mess dvh
        organ_filter = self.hparam.organ_filter 
        for plan in plans:
            for name in plan.dose.copy().keys(): 
                if name not in organ_filter: 
                    plan.dose.pop(name)
                    print(f'pop unnecessary organ for dvh: {name}')

        # plot
        fig, ax = plt.subplots(figsize=(20, 10))
        max_dose = 12000
        organ_names = list(plans[0].dose.keys())
        colors = cm.jet(np.linspace(0,1,len(organ_names)))
        #  linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        linestyles = ['solid', 'dashed', 'dashdot'] 
        if len(plans) > 3: raise NotImplementedError

        for i, organ_name in enumerate(organ_names):
            if self.data.organName_ptsNum[organ_name] != 0:
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
        cprint(f'dvh.pdf has been written to ./tmp.pdf', 'green')

    def run(self):
        plans_to_compare = []
        if self.hparam.NeuralDosePlan:
            plans_to_compare.append(self.load_MonteCarlo_OrganDose(self.mc.old_MUs, 'NeuralDose'))
        if self.hparam.CGDeposPlan:
            plans_to_compare.append(self.load_Depos_OrganDose('CG_depos', scale=self.hparam.CGDeposPlan_doseScale))
        if self.hparam.MCPlan:
            plans_to_compare.append(self.load_MonteCarlo_OrganDose(self.mc.old_MUs, 'CG_MC'))
        if self.hparam.MCMURefinedPlan:
            plans_to_compare.append(self.load_MonteCarlo_OrganDose(unpickle_object(os.path.join(self.hparam.refined_segments_MUs_file,'optimized_MUs.pickle')), 'CG_MC_MURefined'))
        if self.hparam.MCJYPlan:
            plans_to_compare.append(self.load_JYMonteCarlo_OrganDose('CG_JY_MC', '/mnt/win_share2/20200918_NPC_MCDOse_verify_by_JY_congliuReCal/dpm_result*Ave.dat'))
        if self.hparam.TPSFluenceOptimPlan:
            plans_to_compare.append(self.load_TPS_OrganDose('TPSOptim'))
        if self.hparam.FluenceOptimPlan:
            plans_to_compare.append(self.load_fluenceOptim_OrganDose('FluenceOptim'))

        self.comparison_plots(plans_to_compare)

def get_parameters():
    parser = ArgumentParser()

    parser.add_argument('--MCPlan', action="store_true")
    parser.add_argument('--MCJYPlan', action="store_true")
    parser.add_argument('--MCMURefinedPlan', action="store_true")
    parser.add_argument('--deposPlan', action="store_true")

    parser.add_argument('--exp_name', type=str, help='experiment name', required=True)

    parser.add_argument('--device', default='cpu', type=str, help='cpu | cuda')

    hparam, _ = parser.parse_known_args()
    parser.add_argument('--optimized_segments_MUs_file', default='./results/'+hparam.exp_name, type=str, help='will evaluate this optimization results')
    hparam, _ = parser.parse_known_args()

    # tps data parameters
    from data import get_parameters as data_get_parameters 
    data_hparam = data_get_parameters() 
    hparam = Namespace(**vars(hparam), **vars(data_hparam))

    pdb.set_trace()
    if hparam.MCPlan:
        from monteCarlo import get_parameters as gp
        mc_hparam = gp()
        hparam = Namespace(**vars(hparam), **vars(mc_hparam))

    return hparam

if __name__ == "__main__":
    hparam = BaseOptions().parse()
    Evaluation(hparam).run()
