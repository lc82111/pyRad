#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, pdb, glob
from argparse import ArgumentParser, Namespace
from pathlib import Path
from orderedbunch import OrderedBunch


class BaseOptions():
    def __init__(self):
        """This class defines options
        It also implements several helper functions such as parsing, printing, and saving the options.
        It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
        """
        parser = ArgumentParser()

        parser.add_argument('--exp_name', type=str, help='experiment name', required=True)
        parser.add_argument('--patient_ID', type=str, help='gDPM.exe will read and save data to this dir', required=True)

        # TPS parameters: depostion matrix, etc.
        parser.add_argument('--dose_scale', default=1/1000, type=float, help='')
        parser.add_argument('--priority_scale', default=1/100, type=float, help='')
        parser.add_argument('--max_fluence', default=5, type=int, help='max fluence; used to constraint fluence value in fluence optim and plot fluence map.')
        parser.add_argument("--is_check_ray_idx_order", action='store_true', help='')
        parser.add_argument('--MCDose_shape', default='', required=True, type=str, help='to be consistent with gDPM.exe which uses this shape to process dicom.')
        parser.add_argument('--dense_deposition_matrix', action="store_true", help='not use sparse matrix to store deposition matrix')

        # optimization parameters
        parser.add_argument('--optimization_continue', action="store_true")
        parser.add_argument('--steps', default=5000, type=int, help='iter number in fluence optim')
        parser.add_argument('--nb_apertures', default=10, type=int, help='the number of apertures in column generation')
        parser.add_argument('--master_steps', default=5000, type=int, help='total iter in master problem optimization')
        parser.add_argument('--MU_refine_total_steps', default=6000, type=int, help='total iter to run MU optimization')
        parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate')
        parser.add_argument('--plateau_patience', default=50, type=int, help='early stop the optimization when loss does not decrease this times')
        parser.add_argument('--optimizer_name', default='adam', type=str, help='optimizer')
        parser.add_argument('--scheduler_name', default='CosineAnnealingLR', type=str, help='learning rate sheduler')
        parser.add_argument('--device', default='cpu', type=str, help='cpu | cuda; pls use cpu when compute MU*MCDose')
        parser.add_argument('--logs_interval', default=15, type=int, help='log optimization info at interval steps')
        parser.add_argument('--smooth_weight', default=0.2, type=float, help=' fluence smooth regularization weight in fluence optim')

         # Monte Carlo parameters 
        parser.add_argument("--Calculate_MC_unit_doses", action='store_true', help='if true, running FM.exe and gDPM.exe on winServer')
        parser.add_argument('--winServer_nb_threads', default=5, type=int, help='number of threads used for gDPM.exe. Too many threads may overflow the GPU memory')
        parser.add_argument('--test_pbmcDoses', action="store_true", help='if true, plot generated pencilbeam and mc doses')
        parser.add_argument('--test_mcDose', action="store_true", help='if true, plot mc doses')
        parser.add_argument('--mcpbDose2npz', action="store_true", help='if true, calculate and save the pencilbeam mc doses to npz files.')
        parser.add_argument('--mcpbDose2npz_Interp', action="store_true", help='if true, calculate and save the pencilbeam mc doses to npz files.')
        parser.add_argument('--mcpbDose2npz_noRotation_noInterp', action="store_true", help='if true, calculate and save the pencilbeam mc doses to npz files.')
        parser.add_argument('--npz2h5', action="store_true", help='if true, convert npz files to a single h5 file to speed up the training.')
        
        # net param
        parser.add_argument("--net", type=str, default='Unet3D', help="network types: Unet3D|DeepLab3D|DeepLab2D|Unet2D")
        parser.add_argument("--norm_type", type=str, default='GroupNorm', help="batch norm type: GroupNorm|InstanceNorm3d")
        parser.add_argument("--ckpt_path", type=str, default='', help="load check point. Using empty string to avoid the load. Can be used to continue the training.")
        parser.add_argument('--data_dir', type=str, default='', required=True, help='data root for datasets.')
        parser.add_argument("--num_depth", type=int, default=64, help="choose data depth (i.e.num_slices) for 3D net")
        parser.add_argument('--net_output_shape', default='', type=str, help='net output shape')
    
        # evaluation parameters
        parser.add_argument('--TPSFluenceOptimPlan', action="store_true")
        parser.add_argument('--FluenceOptimPlan', action="store_true")
        parser.add_argument('--CGDeposPlan', action="store_true")
        parser.add_argument('--MCPlan', action="store_true")
        parser.add_argument('--MCJYPlan', action="store_true")
        parser.add_argument('--MCMURefinedPlan', action="store_true")
        parser.add_argument('--organ_filter', nargs='+', help='only evaluating these organs')
        parser.add_argument('--CGDeposPlan_doseScale', default=1.0, type=float, help='dose scale for column gen depos plan')
        #organ_filter = ['PTV1-nd2-nx2', 'PTV2', 'PGTVnd', 'PGTVnx', 'Temp.lobe_L', 'Temp.lobe_R', 'Parotid_L', 'Parotid_R', 'Mandible', 'Brainstem+2mmPRV', 'Cord+5mmPRV']
        #organ_filter = ['PTV-smallupper', 'PGTV-plan', 'R1.5', 'Extended_PTV', 'peripheral_tissue']

        self.parser = parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = opt.tensorboard_log
        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options."""
        # parameters depending above 
        hparam = self.parser.parse_args()
        patient_dir = f'../patients_data/{hparam.patient_ID}'
        #  self.parser.add_argument('--deposition_file', default=patient_dir+'/dataset/Deposition_Index.txt', type=str, help='deposition matrix from tps')
        #  self.parser.add_argument('--pointsPosition_file', default=patient_dir+'/dataset/PointsPosition.txt', type=str, help='points positions file')
        #  self.parser.add_argument('--CT_RTStruct_dir', default=patient_dir+'/dataset/CT_RTStruct', type=str, help='ct and rtstruct dicom dir')
        #  self.parser.add_argument('--valid_ray_file', default=patient_dir+'/dataset/ValidMatrix.txt', type=str, help='tps uses this file to indicate the correspondence between the bixels of fluence map and the rays of depostion matrix.')
        #  self.parser.add_argument('--csv_file', default=patient_dir+'/dataset/OrganInfo.csv', type=str, help='this file defines optimization objectives')
        #  self.parser.add_argument('--tps_ray_inten_file', default=patient_dir+'/dataset/TPSray.txt', type=str, help='bixel values optimized by TPS')

        self.parser.add_argument('--optimized_segments_MUs_file_path', default=patient_dir+'/results/'+hparam.exp_name, type=str, help=' optimized seg and MUs saving path in column generation/aperture refine')
        self.parser.add_argument('--optimized_fluence_file_path', default=patient_dir+'/results/'+hparam.exp_name, type=str, help='optimized fluence save path in Fluence Optim')
        self.parser.add_argument('--refined_segments_MUs_file', default=patient_dir+'/results/'+hparam.exp_name, type=str, help='optimized MUs saving path in MU MonteCarlo optim')

        self.parser.add_argument('--deposition_pickle_file_path', default='/mnt/ssd/tps_optimization/'+patient_dir, type=str, help='parsed deposition matrix will be stored in this file')
        self.parser.add_argument('--tensorboard_log', default=patient_dir+'/logs/'+hparam.exp_name, type=str, help='tensorboard directory')
        self.parser.add_argument('--unitMUDose_npz_file', default=patient_dir+'/dataset/MonteCarlo/'+patient_dir+'/unitMUDose.npz', type=str, help='MCDose fetched from winServer will be save in this file')
        self.parser.add_argument('--winServer_MonteCarloDir', default='/mnt/win_share/'+patient_dir, type=str, help='gDPM.exe save MCDose into this directory; this directory is shared by winServer')

        hparam = self.parser.parse_args()
        hparam = vars(hparam) # namespace to dict
        hparam = OrderedBunch(hparam)

        # some file names may vary between patients 
        for k, v in zip(['deposition_file', 'tps_ray_inten_file', 'valid_ray_file', 'csv_file', 'pointsPosition_file', 'MonteCarlo_dir'], \
                        ['Deposition_Index', 'TPSray', 'ValidMatrix', 'OrganInfo', 'PointsPosition.txt', 'Pa*GPU']):
            path = Path(patient_dir, 'dataset', '*'+v+'*')
            fn = glob.glob(str(path))[0]
            hparam[k] = fn
        
        # dicom dir
        DICOM_dir = Path(patient_dir, 'dataset', 'DICOM')
        if DICOM_dir.is_dir():
            hparam.CT_RTStruct_dir = str(DICOM_dir)
        else:
            raise ValueError

        # mc dose shape
        hparam.MCDose_shape = [int(x) for x in hparam.MCDose_shape.split(',')]
        if hparam.net_output_shape != '': 
            hparam.net_output_shape = [int(x) for x in hparam.net_output_shape.split(',')]

        # print all params
        #  self.print_options(hparam)
        return hparam
