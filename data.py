#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
from scipy.sparse import coo_matrix
import numpy.random as npr
import pandas as pd
from skimage.transform import resize

import torch
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter

import os, pprint, pdb, sys, shutil, sys, io, pickle, collections
import SimpleITK as sitk
from argparse import ArgumentParser
from termcolor import colored, cprint
from io import StringIO
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc

from utils import *
pp = pprint.PrettyPrinter(indent=4, width=160)


class Data():
    '''
    A helper class to load Ju Yao data.
    '''
    def __init__(self, hparam):
        self.hparam = hparam
        cprint('get info from csv file.', 'green')
        self._set_paramters_from_csv_table()

        if os.path.isdir(self.hparam.CT_RTStruct_dir):
            cprint('get organ 3D bool index from dicom RTStruct.', 'green')
            self._get_organ_3D_index()
        else:
            cprint('DICOM fold not found.', 'green')
        
        if 'deposition_file' in hparam:
            self.deposition  = self.get_depositionMatrix()
            self.max_ray_idx = self.deposition.shape[1]
            cprint('deposition loaded.', 'green')
        else:
            assert('max_ray_idx' in hparam)
            self.max_ray_idx = hparam.max_ray_idx

        cprint('parsing valid ray data.', 'green')
        self.dict_rayBoolMat, self.dict_rayIdxMat, self.num_beams = self._read_rayIdx_mat() 
        self._set_beamID_rayBeginNum_dict()
        self._set_splitted_depositionMatrix()

    def _set_beamID_rayBeginNum_dict(self):
        '''set dict  {beam_id: (ray_begin_idx, num_rays)}'''
        self.dict_beamID_ValidRayBeginNum = OrderedBunch()
        begin = 0
        for beam_id, mask in self.dict_rayBoolMat.items():
            self.dict_beamID_ValidRayBeginNum[beam_id] = [begin, mask.sum()]  # {beam_id: (ray_begin_idx, num_rays)}
            begin += mask.sum()
        pp.pprint(f'beam_id: (ray_begin_idx, num_rays) {self.dict_beamID_ValidRayBeginNum}')

        # check
        num_bixel = 0
        for beam_id, (_, num) in self.dict_beamID_ValidRayBeginNum.items():
            num_bixel += num
        assert num_bixel == self.deposition.shape[1]

    def get_depositionMatrix(self): 
        if not os.path.isdir(self.hparam.deposition_pickle_file_path):
            os.makedirs(self.hparam.deposition_pickle_file_path)
        fn_depos = os.path.join(self.hparam.deposition_pickle_file_path, 'deposition.pickle') 

        if os.path.isfile(fn_depos):
            cprint('loading deposition data.', 'green')
            D = unpickle_object(fn_depos) 
        else:
            cprint('building deposition data from Deposition_Index.txt', 'green')
            D = self._read_deposition_file_and_save_to_pickle_sparse()  # use sparse matrix to store D

        # check shape
        ptsNum = 0
        for organName, v in self.organ_info.items():
            ptsNum += v['Points Number'] 
        assert ptsNum == D.shape[0], f'shape not match: ptsNum={ptsNum}, deposition_matrix shape={D.shape}'
        print(f'deposition_matrix shape={D.shape}')
        return D

    def _get_uniqueRays_doseGridNum(self):
        print('check ray_index and organs in Deposition_Index.txt')
        ray_list, organName_Dep, doseGridNum = [], [], 0
        is_redundancy = False
        for line in open(self.hparam.deposition_file, 'r'):
            if 'pts_num' in line: # new organ
                pts_num = int(line.split(':')[-1])
                if pts_num == 0:
                    cprint(f'skip organ with zero points, points_num:{pts_num} in Deposition_Index.txt', 'yellow')
                else:
                    organ_name = self.get_organName_from_pointNum(pts_num)
                    if organ_name in organName_Dep:
                        is_redundancy = True
                        cprint(f'skip duplicate points_num:{pts_num} and organ_name:{organ_name} in deposition.txt', 'yellow')
                    else:
                        is_redundancy = False
                        organName_Dep.append(organ_name)
                        cprint(f'find points_num:{pts_num} and organ_name:{organ_name} in deposition.txt', 'green')
            elif 'Indx:' in line:
                ray_list.append(int(line.split(' ')[0].split(':')[-1]))
            elif not is_redundancy and ' ]:' in line:
                doseGridNum += 1

        # ensure deposition_Index.txt has all organs in csv and organs should be consistent with the deposition.txt 
        try:
            if len(self.organ_info.keys()) != len(organName_Dep):
                raise ValueError('organ numbers in Dep and CSV not match')
            for organName_CSV, organName_D in zip(self.organ_info.keys(), organName_Dep):
                if organName_CSV != organName_D:
                    raise ValueError('organ order in Dep and CSV not match')
        except Exception as e:
            print(e)
            pdb.set_trace() # deposition.txt seems lacking organs in cvs.

        # unique ray
        ray_list = list(set(ray_list))
        
        # ensure ray_list shoud == [0, 1, 2, 3, ...]
        if self.hparam.is_check_ray_idx_order:
            for idx in range(len(ray_list)):
              assert idx == ray_list[idx], ray_list[idx]
        else:
            cprint('ray idx order is NOT checked. Some ray idx (should in integer order) may not present in Deposition_Index.txt', 'red')

        return ray_list, doseGridNum

    def _read_deposition_file_and_save_to_pickle_sparse(self):
        '''
        using sparse matrix for deposition matrix to save memory.
        '''
        # get shape inf to build dps matirx
        ray_list, DoseGridNum = self._get_uniqueRays_doseGridNum()

        # fill dps matrix
        print('building depostion matrix')
        row_idxs, col_idxs, values  = [], [], [] 
        with open(self.hparam.deposition_file, "r") as f:
            organ_order, point_idx = [], -1  # organ_order should be consistent with the deposition.txt  
            is_redundancy = False
            for line in f:
                if 'pts_num' in line: # new organ
                    pts_num = float(line.split(':')[-1])
                    if pts_num != 0:
                        organ_name = self.get_organName_from_pointNum(pts_num)
                        if organ_name in organ_order:
                            is_redundancy = True
                        else:
                            is_redundancy = False
                            organ_order.append(organ_name)
                elif not is_redundancy and ' ]:' in line:  # new organ point
                    point_idx += 1
                elif not is_redundancy and 'Indx' in line:  # new ray/bixel
                    ray_idx = int(line.split(' ')[0].split(':')[-1])
                    value   = float(line.split('Pt_dose:')[-1].split('(')[0])
                    row_idxs.append(point_idx)
                    col_idxs.append(ray_idx)
                    values.append(value)

            # build sparse deposition matrix
            D = coo_matrix((values, (row_idxs, col_idxs)), shape=(DoseGridNum, max(ray_list)+1))
            cprint(f'sparse depostion matrix shape: {D.shape}', 'green')
        # save
        pickle_object(os.path.join(self.hparam.deposition_pickle_file_path, 'deposition.pickle'), D)
        return D
    
    def _set_splitted_depositionMatrix(self):
        ''' split depostion matrix, such that a matrix corresponding a beam '''
        cprint('[warning] remove peripheral_tissue dose grid points from deposition matrix', 'red')
        D = self.deposition.tocsr()
        D = D[0:self.get_pointNum_from_organName('ITV_skin')]
        #D = self.deposition.tocsc() # Convert to Compressed Sparse Column format which supports column slice 
        self.dict_beamID_Deps = OrderedBunch()
        for beam_id, (begin, num) in self.dict_beamID_ValidRayBeginNum.items():
            self.dict_beamID_Deps[beam_id] = D[:, begin:begin+num].tocoo() # slice then back to coo format

    def _set_paramters_from_csv_table(self):
        df = pd.read_csv(self.hparam.csv_file, skiprows=1, index_col=0, skip_blank_lines=True)  # duplicated column will be renamed automatically

        # drop nan columns
        cols = [c for c in df.columns if 'Unnamed' not in c] 
        df = df[cols]

        # drop organ with 0 point num
        organ_names = []
        for name, pointNum in df.loc['Points Number'].items():
            if pointNum == '0':
                organ_names.append(name)
        df = df.drop(organ_names, axis='columns')

        # drop another organs if skin present
        is_skin = False
        nonskin_names, skin_names = [], []
        for name in df.columns:
            if 'skin' in name:
                is_skin = True
                skin_names.append(name)
            else:
                nonskin_names.append(name)
        if is_skin:
            self.csv_loss_table = df.drop(skin_names, axis='columns') # this var will be used in loss.py, therefore we should keep the duplicated columns
            df = df.drop(nonskin_names, axis='columns')

        # drop duplicated columns
        df = df.loc[:, ~df.columns.str.replace("(\.\d+)$", "").duplicated()]

        # set up dict of organ info
        self.organ_info = OrderedBunch(df.loc[['Grid Size', 'Points Number']].astype(float).to_dict())
        for organ_name, v in self.organ_info.copy().items():
            self.organ_info[organ_name]['Grid Size'] = v['Grid Size']*10.  # cm to mm
            self.organ_info[organ_name]['Points Number'] = int(v['Points Number'])
        cprint('following csv info will be used to parsing deposition matrix', 'green')
        pp.pprint(dict(self.organ_info))

        tmp = self.csv_loss_table.loc[['Grid Size', 'Points Number', 'Hard/Soft', 'Constraint Type', 'Min Dose', 'Max Dose', 'DVH Volume', 'Priority']]
        cprint('following csv info will be used in loss function', 'green')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(self.csv_loss_table.head(10))

    def get_organName_from_pointNum(self, pointsNum):
        for organName, v in self.organ_info.items():
            if v['Points Number'] == pointsNum:
                return organName
        raise ValueError(f'Can not find organ name with pointNum={pointsNum}')

    def get_pointNum_from_organName(self, organ_name):
        if organ_name not in self.organ_info:
            raise ValueError(f'Can not find organ name in OrganInfo.csv')
        return self.organ_info[organ_name]['Points Number']

    def _read_rayIdx_mat(self):
        # get bool matrixes, where 1 indicates the present of ray
        with open(self.hparam.valid_ray_file, "r") as f:
            dict_rayBoolMat = collections.OrderedDict() 
            beam_id = 0
            for line in f:
                if 'F' in line: # new beam 
                    beam_id = int(line.replace('F','')) + 1 # NOTE: index of beam start from 1
                    dict_rayBoolMat[beam_id] = [] 
                else:
                    row = np.loadtxt(StringIO(line))
                    dict_rayBoolMat[beam_id].append(row)
        num_beams = beam_id
        
        # convert (list of 1D arrays) to (2D matrix)
        ray_num = 0
        for beam_id, FM in dict_rayBoolMat.copy().items():
            FM = np.asarray(FM, dtype=np.bool)
            dict_rayBoolMat[beam_id] = FM
            ray_num += FM.sum()
        assert ray_num == self.max_ray_idx
        assert ray_num == self.deposition.shape[1], f'shape not match: rayNum={ray_num}, deposition_matrix shape={D.shape}'

        # convert 1 in bool matrixes to ray idx
        dict_rayIdxMat = collections.OrderedDict()
        ray_idx = -1
        for beam_id, FM in dict_rayBoolMat.items():
            idx_matrix = np.full_like(FM, self.max_ray_idx, dtype=np.int)  # NOTE: using max_ray_idx to indicate non-valid ray 
            for row in range(FM.shape[0]):
                for col in range(FM.shape[1]):
                    if FM[row, col] == 1:
                        ray_idx += 1
                        idx_matrix[row, col] = ray_idx
            dict_rayIdxMat[beam_id] = idx_matrix
        return dict_rayBoolMat, dict_rayIdxMat, num_beams

    def project_to_fluenceMaps(self, fluenceVector):
        '''Convert 1D fluenceVector to 2D fluenceMap
            Arguments: fluenceVector: ndarray (#bixels, )
            Return: {beam_id: fluenceMap ndarray (H,W)} ''' 
        # set up a tmp with shape:(#bixels+1, ) and tmp[#bixels+1]=0; 
        # where #bixels+1 indicate non-valid ray.
        # In this way, we can set the intensity of nonvalid ray to 0.
        tmp = np.append(fluenceVector, 0)
        dict_FluenceMap = collections.OrderedDict()
        # construct 2D fluence matrix from fluenceVector using numpy's fancy 2D indice
        for beam_id, ray_idx in self.dict_rayIdxMat.items():
            dict_FluenceMap[beam_id] = tmp[ray_idx]
        return dict_FluenceMap

    def project_to_fluenceMaps_torch(self, fluence):
        '''fluence: (#bixels, )
            return: {beam_id: fluenceMap with the shape of (H,W)}
        ''' 
        # set up a tmp with shape:(#bixels+1, ) and tmp[#bixels+1]=0; 
        tmp = torch.cat([fluence, torch.tensor([0.,], dtype=torch.float32, device=fluence.device)]) # shape:(max_ray_idx, ); tmp[max_ray_idx]=0
        dict_FluenceMap = collections.OrderedDict()
        for beam_id, ray_idx in self.dict_rayIdxMat.items():
            dict_FluenceMap[beam_id] = tmp[ray_idx]
        return dict_FluenceMap

    def get_rays_from_fluences(self, dict_FluenceMat):
        '''dict_fluences: {beam_id: fluence matrix}
        return:
            valid_rays: (#valid_bixels,)
        '''
        valid_rays = []
        for (_, boolMat), (idx, F) in zip(self.dict_rayBoolMat.items(), dict_FluenceMat.items()):
            valid_rays.append(F[boolMat].flatten())
        valid_rays = np.concatenate(valid_rays, axis=0)
        return valid_rays

    def project_to_validRays_torch(self, dict_fluences):
        ''' Convert flatten fluenceMap to valid fluenceVector
        Arguments: 
            dict_fluences: {beam_id: fluence vector}
        Return:
            valid_rays: (#valid_bixels,)
            dict_fluenceMaps: {beam_id: fluence matrix} '''
        dict_fluenceMaps = OrderedBunch()
        valid_rays = []
        for (beam_id, msk), (_, fluence) in zip(self.dict_rayBoolMat.items(), dict_fluences.items()):
            msk = torch.tensor(msk, dtype=torch.bool, device=fluence.device)
            valid_rays.append(fluence.view(*msk.shape)[msk].flatten()) # select valid rays and back to 1d vector
            dict_fluenceMaps[beam_id] = fluence.detach()
        valid_rays = torch.cat(valid_rays, axis=0)
        return valid_rays, dict_fluenceMaps

    def _get_organ_3D_index(self):
        '''
        Return: self.organ_masks {organ_name: bool mask (z=167, x=512, y=512)}
        '''
        ## get organ priorities from csv file
        df = self.csv_loss_table

        # only consider min_dose and priority
        df = df.loc[['Min Dose','Priority']]

        # string to float
        df = df.astype(float)

        # add a row to indentify ptv/oar
        ptv_oar = [1 if 'TV' in name else 0 for name in df.columns]
        ptv_oar = np.array(ptv_oar).reshape(1,-1)
        names = [name for name in df.columns]
        df2 = pd.DataFrame(ptv_oar, index=['ptv/oar'], columns=names)
        df = df.append(df2)
        df = df.loc[:, ~df.columns.str.replace("(\.\d+)$", "").duplicated()] # remove deuplicated organ name

        # sort to identify the overlapped organs and write to dataset dir to verify
        sorted_df = df.sort_values(by=['Priority', 'ptv/oar', 'Min Dose'], axis='columns', ascending=False)
        sorted_df.to_csv(self.hparam.csv_file.replace('OrganInfo.csv', 'sorted_organs.csv'))
        cprint('following organ order will be used to parse RTStruct', 'green')
        print(sorted_df)
       
        ## get contour from dicom

        # ensure all organ_names in csv appeared in RTStruct
        Dicom_Reader = Dicom_to_Imagestack(get_images_mask=True, arg_max=True)  # arg_max is important to get the right order for overlapped organs.
        Dicom_Reader.Make_Contour_From_directory(self.hparam.CT_RTStruct_dir)
        roi_names = []
        is_rtstruct_complete = True
        for name in sorted_df.columns:
            if name not in Dicom_Reader.all_rois:
                cprint(f'Warning: {name} not in RTStruct! we simply skip it.', 'red')
                is_rtstruct_complete == False
            else:
                roi_names.append(name)
        cprint(f'number of organ: {len(roi_names)}', 'green')
        if not is_rtstruct_complete:
            raise ValueError('some organ not in RTStruct')

        # get contours 
        Dicom_Reader.set_contour_names(roi_names)
        Dicom_Reader.Make_Contour_From_directory(self.hparam.CT_RTStruct_dir)
        
        # match MonteCarlo dose's shape 
        if Dicom_Reader.mask.shape != self.hparam.MCDose_shape:
            cprint(f'\nresize contour {Dicom_Reader.mask.shape} to match MC shape {self.hparam.MCDose_shape}', 'yellow')
            Dicom_Reader.mask = resize(Dicom_Reader.mask, self.hparam.MCDose_shape, order=0, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        # match network output shape
        if self.hparam.net_output_shape != '':
            if Dicom_Reader.mask.shape[0] != self.hparam.net_output_shape[0]: 
                cprint(f'resize and crop contour {Dicom_Reader.mask.shape} to match network output shape {self.hparam.net_output_shape}', 'yellow')
                Dicom_Reader.mask = resize(Dicom_Reader.mask, (self.hparam.net_output_shape[0],)+Dicom_Reader.mask.shape[1:], \
                                           order=0, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                crop_top  = int((self.hparam.MCDose_shape[1]-self.hparam.net_output_shape[1] + 1) * 0.5)
                crop_left = int((self.hparam.MCDose_shape[2]-self.hparam.net_output_shape[2] + 1) * 0.5)
                Dicom_Reader.mask = Dicom_Reader.mask[:, crop_top:crop_top+self.hparam.net_output_shape[1], crop_left:crop_left+self.hparam.net_output_shape[2]]

        cprint(f'shape of contour label volume = {Dicom_Reader.mask.shape}', 'green')
        cprint(f'max label in contour label volume = {Dicom_Reader.mask.max()}', 'green')

        # label mask -> bool mask
        self.organ_masks = OrderedBunch()
        for i in range(1, Dicom_Reader.mask.max()+1): # iter over contours
            tmp = np.zeros_like(Dicom_Reader.mask, dtype=np.bool) 
            tmp[Dicom_Reader.mask==i] = True
            self.organ_masks[roi_names[i-1]] = tmp

        # we may use these var out the method 
        self.CT = Dicom_Reader.dicom_handle
        self.Dicom_Reader = Dicom_Reader

        debug = False 
        if debug:  # show overlapped ct
            pdb.set_trace() 
            os.environ['SITK_SHOW_COMMAND'] = '/home/congliu/Downloads/Slicer-4.10.2-linux-amd64/Slicer'
            dicom_handle = Dicom_Reader.dicom_handle
            #annotations_handle = sitk.GetImageFromArray(self.organ_masks['Brainstem+2mmPRV'])
            #  annotations_handle = sitk.GetImageFromArray(self.organ_masks['Brainstem+2mmPRV', 'PTV1-nd2-nx2', 'PTV2'])
            #  annotations_handle = sitk.GetImageFromArray(self.organ_masks['PTV1-nd2-nx2'])
            annotations_handle = sitk.GetImageFromArray(self.organ_masks['Parotid_L'])
            annotations_handle.CopyInformation(dicom_handle)
            overlay = sitk.LabelOverlay(dicom_handle, annotations_handle, 0.1)
            sitk.Show(overlay)

class Geometry():
    def __init__(self, data):
        self.set_CT(data)
        self.set_plan(data)
        self.set_doseGrid(data)
        self.prints()
    
    def prints(self):
        pp.pprint(f'CT info:       {dict(self.CT)}')
        print()
        for k, v in self.plan.items():
            if 'beam_info' in k:
                for _k, _v in v.items():
                    pp.pprint(f'{_k} info : {dict(_v)}')
            else:
                pp.pprint(f'{k} info : {v}')
        print()
        pp.pprint(f'doseGrid info:     {dict(self.doseGrid)}')
        print()

    def set_CT(self, data):
        self.CT = OrderedBunch({'spacing': np.array(data.Dicom_Reader.dicom_handle.GetSpacing(), dtype=np.float32), # [1.171875, 1.171875, 2.5]mm@512x512x126, 
                                'size':    np.array(data.Dicom_Reader.dicom_handle.GetSize(), dtype=np.int), # (512,512,126)
                                'origin':  np.array(data.Dicom_Reader.dicom_handle.GetOrigin()), # [-300, -300, z]mm
                                 })

    def set_plan(self, data):
        beam_info = data.Dicom_Reader.beam_info # isocenter: [-18.09999, -1.599998, 246.3]mm
        self.plan = OrderedBunch({'isocenter': beam_info[1].IsoCenter, 
                                  'beam_numbers': len(beam_info),
                                  'beam_info': beam_info,
                                 })

    def get_isoCenter_in_pixelCoordinates256x256(self):
        spacing = self.CT.spacing * 2    # [2.34375, 2.34375, 5]mm@256x256x63
        isocenter_mm = self.plan.isocenter  # [-18.09999, -1.599998, 246.3]mm

        pixel_isocenter = (isocenter_mm[0:2] - self.CT.origin[0:2]) / spacing[0:2]  # [120.2773376  128.68266581]
        pixel_isocenter[1] = 256 - pixel_isocenter[1]                                 # move the origin from left bottom to left top 
        cprint(f'pixel_isocenter={pixel_isocenter}', 'green')

    def set_doseGrid(self, data):
        gs = data.organ_info['ITV_skin']['Grid Size']
        doseGrid_spacing = np.array([gs, gs, 2.5]) # juyao give 2.5 to me
        cprint(f'using juyao given z spacing 2.5 !','red')
        self.doseGrid = OrderedBunch({'spacing': doseGrid_spacing,
                                      'size': (self.CT.size * self.CT.spacing / doseGrid_spacing).astype(np.int),
                                      })
