#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr
import pandas as pd
from termcolor import cprint
from dicompylercore import dicomparser, dvh, dvhcalc
from skimage.transform import resize, rotate 

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter

import os, pdb, sys, collections, pickle, shutil, glob, pydicom
from argparse import ArgumentParser, Namespace
from termcolor import colored, cprint
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc

from utils import *
from data import Data, Geometry
from loss import Loss
from options import BaseOptions
from monteCarlo import MonteCarlo
from neural_dose import NeuralDose
from pencil_beam import PencilBeam
from neuralDose.utils import gamma_plot, MyModelCheckpoint
from neuralDose.metrics import Gamma


class Evaluation():
    def __init__(self, hparam):
        self.hparam = hparam
        # init data and loss
        self.data = Data(hparam)
        self.loss = Loss(hparam, self.data.allOrganTable)
        self.geometry = Geometry(self.data)
        # deposition matrix (#doseGrid, #bixels)
        self.deposition = convert_depoMatrix_to_tensor(self.data.deposition, self.hparam.device)
        # gamma setting
        D,H,W = self.hparam.net_output_shape
        self.gm = Gamma(gDPM_config_path=f'/mnt/win_share/{self.hparam.patient_ID}/templates/gDPM_config.json', shape=(H,W,D))
        # prescription_dose
        self.prescription_dose = self.geometry.plan.target_prescription_dose
        # results dir
        time_str = get_now_time()
        self.save_path = Path(self.hparam.evaluation_result_path, time_str)
        make_dir(self.save_path)
        # ptv mask
        self.ptv_mask = self.data.organ_masks[self.hparam.PTV_name]

        # MC dose
        if hparam.MCPlan or hparam.MCJYPlan or hparam.MCMURefinedPlan or hparam.NeuralDosePlan or hparam.PBPlan or hparam.originalPlan:
            self._set_optimized_segments_MUs()
            self.mc = MonteCarlo(hparam, self.data)
            self.unitDose = self.get_all_beams_MCdose_for_optimizedSegMUs()  # unitDose, ndarray (nb_beams*nb_apertures=30, D/2=61, H=CenterCrop128, W=CenterCrop128)
        
        if hparam.PBPlan or hparam.NeuralDosePlan or hparam.originalPlan:
            self.PB = PencilBeam(hparam, self.data)
            self.neuralDose = NeuralDose(hparam, self.data, self.PB)

    def _set_optimized_segments_MUs(self):
        ''' Return: 
                self.segs_mus: optimized segments and MUs {beam_id: {'Seg':Segs ndarray (hxw, nb_apertures), 'MU':MUs ndarray (nb_apertures,)}} 
                self.optimized_MUs:  optimized MUs  ndarray (nb_beams*nb_apertures, 1, 1, 1) 
                self.optimized_segs: optimized segs dict {beam_id: segs ndarray (nb_apertures, h, w)}, hxw==#bixels 
                self.nb_apertures, self.nb_beams: int.
        '''
        file_name = os.path.join(self.hparam.optimized_segments_MUs_file_path, 'optimized_segments_MUs.pickle')
        with open(file_name, 'rb') as f:
            self.segs_mus = pickle.load(f)

        self.nb_beams     = len(self.segs_mus)
        self.nb_apertures = len(self.segs_mus[1]['MU']) 

        self.optimized_MUs = np.empty((self.nb_beams*self.nb_apertures, 1, 1, 1), dtype=np.float32)
        self.optimized_segs = OrderedBunch()
        for beam_id, seg_mu in self.segs_mus.items():
            H, W = self.data.dict_bixelShape[beam_id]  # (h,w), hxw==#bixels
            segs, mus = seg_mu['Seg'], seg_mu['MU']
            self.optimized_MUs[(beam_id-1)*self.nb_apertures : (beam_id-1)*self.nb_apertures+self.nb_apertures] = mus.reshape((self.nb_apertures,1,1,1))
            self.optimized_segs[beam_id] = segs.T.reshape(self.nb_apertures, H, W)

    def _get_gamma(self, mc_dose, dict_doses):
        ret = {} 
        path = Path(self.hparam.evaluation_result_path)
        for name, dose in dict_doses.items():
            gamma_path = path.joinpath(f'{name}_gammaArr.pickle')
            csv_path   = path.joinpath(f'{name}_gammaCSV.csv')
            if not os.path.isfile(gamma_path):
                passRate, gammaArr = self.gm.get_a_gamma(dose_ref=to_np(mc_dose), dose_pred=to_np(dose))
                pickle_object(gamma_path, gammaArr)
                df = pd.DataFrame({'passRate':[passRate,]})
                df.to_csv(csv_path)
            else:
                gammaArr = unpickle_object(gamma_path)
            ret[name] = gammaArr
        return ret
   
    def get_all_beams_MCdose_for_optimizedSegMUs(self):
        ''' Return: unit dose, ndarray (#beams*#apertures, D/2, H=centerCrop128, W=centerCrop128) '''
        if os.path.isfile(self.hparam.unitMUDose_npz_file):
            cprint(f'load {self.hparam.unitMUDose_npz_file}', 'green')
            return np.load(self.hparam.unitMUDose_npz_file)['npz']

        else:
            # calculate mc dose on winServer
            if not Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results', f'dpm_result_{self.nb_beams}_{self.nb_apertures-1}Ave.dat').is_file(): 
                self.mc.cal_unit_MCdose_on_winServer(self.optimized_segs)
            else:
                cprint(f'[Warning], use saved gDPM results on local disk: {self.hparam.unitMUDose_npz_file}', 'red')
            
            # fetch mc dose from winServer
            cprint(f'fetching gDPM results from winServer and save them to local disk: {self.hparam.unitMUDose_npz_file}', 'green')
            if self.hparam.net_output_shape != '': 
                shape = self.hparam.net_output_shape
            else:
                shape = self.hparam.MCDose_shape
            unit_dose = np.empty((self.nb_beams*self.nb_apertures, ) + shape, dtype=np.float32)
            idx = 0
            for beam_id in range(1, self.nb_beams+1):
                for aperture_id in range(0, self.nb_apertures):
                    unit_dose[idx] = self.mc.get_unit_MCdose_from_winServer(beam_id, aperture_id)
                    idx += 1
            np.savez(self.hparam.unitMUDose_npz_file, npz=unit_dose)
            
            # del gDPM results on winServer to aviod misuse another experiments' gDPM results
            dpm_result_path = Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results')
            del_fold(dpm_result_path )
            make_dir(dpm_result_path )
            return unit_dose

    def gamma_plot_neuralDose(self, mc_dose, pb_dose, nd_dose, is_plot=True):
        gammaArr = self._get_gamma(mc_dose, {'pb':pb_dose, 'nd':nd_dose})
        if is_plot:
            fig_save_path = self.save_path.joinpath(f'total_dose_neuralDose')
            make_dir(fig_save_path)
            gamma_plot(self.neuralDose.CTs, self.ptv_mask, to_np(mc_dose), to_np(pb_dose), to_np(nd_dose), gammaArr['pb'], gammaArr['nd'], fig_save_path, self.prescription_dose)
        cprint(f'gamma_plot done. saved to{fig_save_path}', 'green')

    def gamma_plot_PB(self, is_plot=True):
        mc_dose = self.load_MonteCarlo_OrganDose(self.optimized_MUs, 'MonteCarloDose')['skin_dose']
        pb_dose = self.load_penceilBeamDose_OrganDose('PB')['skin_dose']
        mask = self.data.organ_masks[self.hparam.PTV_name]
        gammaArr = self._get_gamma(mc_dose, {'pb':pb_dose})
        if is_plot:
            fig_save_path = self.save_path.joinpath(f'total_dose_PB')
            make_dir(fig_save_path) 
            gamma_plot(self.neuralDose.CTs, self.ptv_mask, to_np(mc_dose), to_np(pb_dose), to_np(pb_dose), gammaArr['pb'], gammaArr['pb'], fig_save_path, self.prescription_dose)
        cprint(f'gamma_plot done. saved to{fig_save_path}', 'green')

    def gamma_plot_original(self, is_plot=True):
        mc_dose = self.load_originalMC_OrganDose('pbMonteCarloDose')['skin_dose']
        pb_dose = self.load_originalPB_OrganDose('pbDose')['skin_dose']
        mask = self.data.organ_masks[self.hparam.PTV_name]

        D,H,W = self.hparam.net_output_shape
        gamma = Gamma(gDPM_config_path=f'/mnt/win_share/{self.hparam.patient_ID}/templates/gDPM_config.json', shape=(H,W,D))

        pb_pass, pbGamma = gamma.get_a_gamma(dose_ref=to_np(mc_dose), dose_pred=to_np(pb_dose))
        if is_plot:
            fig_save_path = self.save_path.joinpath(f'total_dose_original')
            make_dir(fig_save_path)
            gamma_plot(self.neuralDose.CTs, self.ptv_mask, to_np(mc_dose), to_np(pb_dose), to_np(pb_dose), pbGamma, pbGamma, fig_save_path, self.prescription_dose)

        cprint(f'gamma_plot done. saved to{fig_save_path}', 'green')

    def load_MonteCarlo_OrganDose(self, MUs, name, scale=1):
        ''' self.optimized_MUs: ndarray (#beams*#apertures, 1, 1, 1)  '''
        MUs = np.abs(MUs) / self.hparam.dose_scale  # x1000
        dose = self.unitDose * MUs * scale  # unitDose, ndarray (nb_beams*nb_apertures=30, D/2=61, H=CenterCrop128, W=CenterCrop128) 
        dose = dose.sum(axis=0, keepdims=False)  #  (D, H, W) 
        dose = torch.tensor(dose, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(dose, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'organ_dose':dict_organ_doses, 'skin_dose':dose, 'name':name})

    def load_NeuralDose_OrganDose(self, name):
        cprint(f'neuralDose Plan uses following ptimized_segments_MUs:{self.hparam.optimized_segments_MUs_file_path}; {self.hparam.deposition_pickle_file_path}', 'yellow')

        # cal neural dose
        neural_dose = 0
        for beam_id, segs_mus in self.segs_mus.items(): # for each beam
            mask = self.data.dict_rayBoolMat_skin[beam_id]   # (h, w), hxw==#bixels
            segs, mus = segs_mus['Seg'], segs_mus['MU'] # (#bixels, #apertures), (#apertures)
            mus  = mus / self.hparam.dose_scale  # x1000
            segs = torch.tensor(segs, dtype=torch.float32, device=self.hparam.device)
            mus  = torch.tensor(mus,  dtype=torch.float32, device=self.hparam.device)
            mask = torch.tensor(mask, dtype=torch.bool,    device=self.hparam.device)
            neural_dose += self.neuralDose.get_neuralDose_for_a_beam(beam_id, mus, segs, mask)
            assert tuple(neural_dose.shape) == self.hparam.net_output_shape

        # get individual organ doses
        dict_organ_doses   = parse_MonteCarlo_dose(neural_dose, self.data)

        return OrderedBunch({'name': name, 'organ_dose':dict_organ_doses, 'skin_dose': neural_dose})

    def load_penceilBeamDose_OrganDose(self, name):
        cprint(f'pencilBeam Plan uses following ptimized_segments_MUs:{self.hparam.optimized_segments_MUs_file_path}; {self.hparam.deposition_pickle_file_path}', 'yellow')

        # compute neural dose
        pb_dose = 0
        for beam_id, segs_mus in self.segs_mus.items(): # for each beam
            mask = self.data.dict_rayBoolMat_skin[beam_id]   # (h, w), hxw==#bixels
            segs, MUs = segs_mus['Seg'], segs_mus['MU'] # (#bixels, #apertures), (#apertures)
            MUs  = MUs / self.hparam.dose_scale  # x1000
            segs = torch.tensor(segs, dtype=torch.float32, device=self.hparam.device)
            MUs  = torch.tensor(MUs,  dtype=torch.float32, device=self.hparam.device)
            mask = torch.tensor(mask, dtype=torch.bool,    device=self.hparam.device)
            for i in range(segs.shape[-1]): # for each aperture
                seg = segs[:,i][mask.flatten()]
                unitDose = self.PB.get_unit_pencilBeamDose(beam_id, seg)  # 3D dose 61x128x128
                pb_dose += unitDose * MUs[i]  # x MU
            assert tuple(pb_dose.shape) == self.hparam.net_output_shape

        # get individual organ doses
        dict_organ_doses  = parse_MonteCarlo_dose(pb_dose, self.data)

        return OrderedBunch({'name': name, 'organ_dose':dict_organ_doses, 'skin_dose': pb_dose})

    def load_originalMC_OrganDose(self, name):
        fns = glob.glob(f'{self.hparam.DICOM_dir}/RTDOSE*原计划.dcm')
        assert len(fns) == 1
        cprint(f'loading original mc dose from {fns[0]}', 'yellow')

        dose = load_DICOM_dose(fns[0], self.geometry, self.hparam.MCDose_shape)
        dose = torch.tensor(dose, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(dose, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'organ_dose':dict_organ_doses, 'skin_dose':dose, 'name':name})

    def load_originalPB_OrganDose(self, name):
        fns = glob.glob(f'{self.hparam.DICOM_dir}/RTDOSE*原计划-good.dcm')
        assert len(fns) == 1
        cprint(f'loading original pb dosefrom {fns[0]}', 'yellow')

        dose = load_DICOM_dose(fns[0], self.geometry, self.hparam.MCDose_shape)
        dose = torch.tensor(dose, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(dose, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'organ_dose':dict_organ_doses, 'skin_dose':dose, 'name':name})

    def load_JYMonteCarlo_OrganDose(self, name, dosefilepath, scale=1):
        MCdoses = self.mc.get_JY_MCdose(dosefilepath) * scale
        MCdoses = torch.tensor(MCdoses, dtype=torch.float32, device=self.hparam.device)
        dict_organ_doses = parse_MonteCarlo_dose(MCdoses, self.data)  # parse organ_doses to obtain individual organ doses
        return OrderedBunch({'organ_dose':dict_organ_doses, 'name':name})

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
        
        return OrderedBunch({'fluence':to_np(fluence), 'organ_dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def load_TPS_OrganDose(self, name='TPSOptimResult'):
        # intensity
        fluence = np.loadtxt(self.hparam.tps_ray_inten_file)
        fluence = torch.tensor(fluence, dtype=torch.float32, device=self.hparam.device)
        dict_FMs = self.data.project_to_fluenceMaps(to_np(fluence))

        # dose
        doses = cal_dose(self.deposition, fluence)
        # split organ_doses to obtain individual organ doses
        dict_organ_doses = split_doses(doses, self.data.organName_ptsNum)

        return OrderedBunch({'fluence':to_np(fluence), 'organ_dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

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
        
        return OrderedBunch({'fluence':to_np(fluence), 'organ_dose':dict_organ_doses, 'fluenceMaps': dict_FMs, 'name': name})

    def comparison_plots(self, plans, dvh_name=''):
        '''
        plans: list of plan
        '''
        ## print loss and breaking pts num
        for plan in plans: 
            dict_organ_doses = plan.organ_dose.copy()
            for name, dose in dict_organ_doses.copy().items():
                dict_organ_doses.update({name: dose*self.hparam.dose_scale})  # /1000
            plan_loss, plan_breaking_points_nums = self.loss.loss_func(dict_organ_doses)
            print(f'{plan.name} breaking points #: ', end='')
            for organ_name, breaking_points_num in plan_breaking_points_nums.items():
                print(f'{organ_name}: {breaking_points_num}   ', end='')
            print(f'loss={plan_loss}\n\n') # NOTE: the loss may differ from the training loss because the partialExp effects

        ## plot DVH
        # pop unnecessary organ dose to avoid mess dvh
        for plan in plans:
            for name in plan.organ_dose.copy().keys(): 
                if name not in self.hparam.consider_organs: 
                    plan.organ_dose.pop(name)
                    print(f'DVH: pop unnecessary organ: {name}')
        # plot
        fig, ax = plt.subplots(figsize=(5, 5))
        organ_names = list(plans[0].organ_dose.keys())
        colors = cm.jet(np.linspace(0,1,len(organ_names)))
        #  linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        linestyles = ['solid', 'dashed', 'dashdot'] 
        if len(plans) > 3: raise NotImplementedError

        for i, organ_name in enumerate(organ_names):
            if self.data.get_pointNum_from_organName(organ_name) != 0:
                for pi, plan in enumerate(plans):
                    n, bins, patches = ax.hist(to_np(plan.organ_dose[organ_name]),
                       bins=12000, 
                       linestyle=linestyles[pi], color=colors[i],
                       range=(0, self.prescription_dose*1.3),
                       density=True, histtype='step',
                       cumulative=-1, 
                       #label=f'{plan.name}_{organ_name}_maxDose{int(to_np(plan.organ_dose[organ_name].max()))}')
                       label=f'{plan.name}_{organ_name}')

        ax.grid(True)
        #ax.legend(loc='upper left', bbox_to_anchor=(1.05,1.0))
        ax.legend(loc='best', fontsize='x-small')
        ax.set_title('Dose volume Histograms')
        ax.set_xlabel('Absolute Dose cGy')
        ax.set_ylabel('Relative Volume %')
        plt.tight_layout()
        save_path = self.save_path.joinpath(f'./DVH_{dvh_name}.pdf')
        plt.savefig(save_path)
        #plt.show()
        cprint(f'dvh_{dvh_name}.pdf has been written to {save_path}', 'green')

    def run(self):
        plans_to_compare = []
        # PB dose 
        if self.hparam.PBPlan:
            plans_to_compare.append(self.load_penceilBeamDose_OrganDose('PB'))
            plans_to_compare.append(self.load_MonteCarlo_OrganDose(self.optimized_MUs, 'PBMC'))
            self.gamma_plot_PB()
        # neural dose 
        if self.hparam.NeuralDosePlan:
            mc_doses = self.load_MonteCarlo_OrganDose(self.optimized_MUs, 'NeuMC')
            pb_doses = self.load_penceilBeamDose_OrganDose('PB')
            nd_doses = self.load_NeuralDose_OrganDose('Neu')

            self.comparison_plots([mc_doses, pb_doses], dvh_name='mc_pb')
            self.comparison_plots([mc_doses, nd_doses], dvh_name='mc_nd')

            self.gamma_plot_neuralDose(mc_doses['skin_dose'], pb_doses['skin_dose'], mc_doses['skin_dose'])
            return
        # original dose
        if self.hparam.originalPlan:
            plans_to_compare.append(self.load_originalMC_OrganDose('originalMC'))
            plans_to_compare.append(self.load_originalPB_OrganDose('originalPB'))
            self.gamma_plot_original()

        if self.hparam.CGDeposPlan:
            plans_to_compare.append(self.load_Depos_OrganDose('CG_depos', scale=self.hparam.CGDeposPlan_doseScale))
        if self.hparam.MCMURefinedPlan:
            plans_to_compare.append(self.load_MonteCarlo_OrganDose(unpickle_object(os.path.join(self.hparam.refined_segments_MUs_file,'optimized_MUs.pickle')), 'CG_MC_MURefined'))
        if self.hparam.MCJYPlan:
            plans_to_compare.append(self.load_JYMonteCarlo_OrganDose('CG_JY_MC', '/mnt/win_share2/20200918_NPC_MCDOse_verify_by_JY_congliuReCal/dpm_result*Ave.dat'))
        if self.hparam.TPSFluenceOptimPlan:
            plans_to_compare.append(self.load_TPS_OrganDose('TPSOptim'))
        if self.hparam.FluenceOptimPlan:
            plans_to_compare.append(self.load_fluenceOptim_OrganDose('FluenceOptim'))

        self.comparison_plots(plans_to_compare)


if __name__ == "__main__":
    hparam = BaseOptions().parse()
    Evaluation(hparam).run()
