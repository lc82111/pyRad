#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
#  matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pydicom, h5py
from braceexpand import braceexpand
from scipy.interpolate import interpn, griddata
from skimage.transform import resize, rotate 
from skimage.draw import set_color, random_shapes
from skimage.exposure import rescale_intensity
import numpy as np
from scipy.ndimage.measurements import label 
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr

import torch, torchvision
from torch import nn
import torch.nn.functional as torchF
from torch.utils.tensorboard import SummaryWriter
import webdataset as wds

from argparse import ArgumentParser
from termcolor import colored, cprint
from io import StringIO
import sys, glob, os, pdb, shutil, pickle, collections, time
from pathlib import Path
from socket import *

#sys.path.insert(0, '/home/congliu/lina_tech/tps_optimization/FluenceMap/codes')
from utils import *
from data import Data, Geometry 
from options import BaseOptions
from multiprocessing import Process 

def call_FM_gDPM_on_windowsServer(PID, nb_beams, nb_apertures, nb_threads):
    cprint(f'send msg to windows Server to call FM.exe and gDPM.exe.', 'green')

    host = "192.168.10.103" # set to IP address of target computer
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
        cprint("winServer say mission completed ", 'green')
    except KeyboardInterrupt:
        print('ctl-c pressed; exit.....')
        UDPSock.close()
        os._exit(0)
    except Exception as e:
        cprint(f"Error in call_FM_gDPM_on_windowsServer: {e}", 'red')
        os._exit(0)

class PencilBeam():
    def __init__(self, hparam, data, roi_skinName='ITV_skin'):
        self.hparam = hparam
        self.roi_skinName = roi_skinName 
        self.data = data
        self.geometry = Geometry(data)
        self.dict_randomApertures = None

        # set dose grid coords
        self._set_points_positions()

        # set corner of maximum skine rect
        self._set_skin_coords()
        
    def _load_randApert(self):
        if self.dict_randomApertures == None:
            # load saved random apertures
            save_path = Path(self.hparam.patient_ID).joinpath('dataset/dict_randomApertures.pickle')
            if os.path.isfile(save_path):
                self.dict_randomApertures = unpickle_object(save_path)
            else:
                raise ValueError

    def _set_skin_coords(self): 
        '''get corner of maximum skin rectangle  '''
        dose_grid = self.dose_grid[:, 0:2]
        dose_grid = dose_grid.round().astype(np.uint)

        self.skin_lefttop = OrderedBunch({'x':dose_grid[:,0].min(), 'y':dose_grid[:,1].min()})
        self.skin_rightbot= OrderedBunch({'x':dose_grid[:,0].max(), 'y':dose_grid[:,1].max()})

    def _set_points_positions(self):
        '''get dose_grid coords, from pointsPosition.txt
        return: coords (#points, 3=[x,y,z])
        '''
        with open(self.hparam.pointsPosition_file, "r") as f:
            lines = f.readlines()
        
        # get begin and end lines in pointsPosition.txt for roi_skin
        begin, end = np.inf, np.inf
        for i, line in enumerate(lines):
            if self.roi_skinName in line: 
                begin = i+1
            if i>=begin and ':' in line: # another organ 
                end = i
            if i>=begin and i+1==len(lines):
                end = i+1
        
        # parse each line for x,y,z coords
        coords = []
        lines = lines[begin:end]
        for line in lines:
            coord = [float(x) for x in line.split()]  # split uses dafault delimiter: space and \n 
            coords.append(coord)
        coords = np.asarray(coords)
        assert len(coords) == self.data.get_pointNum_from_organName(self.roi_skinName)

        # sort pointPositions to ensure the consistent ordering with deposition.txt: z decending, y decending, x ascending  
        sort_index = np.lexsort((coords[:,0], -coords[:,1], -coords[:,2]))  # NOTE TODO: the PTVs have duplicate begin and end slices
        coords = coords[sort_index]
        
        # change zs from physic coords (e.g. 3.6) to image coords (e.g. 0)
        zs = np.array([float(z)/10 for z in self.data.Dicom_Reader.slice_info])  # z axis coords in the physic coords 
        for i, z in enumerate(zs): # for each slice
            indexes = np.where(coords[:,-1]==z)[0]
            coords[indexes, -1] = i

        # for later use
        self.dose_grid = coords
        self.doseGrid_zz_yy_xx = (coords[:,2], coords[:,1], coords[:,0])

    def _parse_dose_torch(self, vector_dose):
        ''' turn 1D dose to 3D and interp the 3D dose
        return: 3D dose (1, 1, #slice, H=256, W=256) '''
        D, H, W = self.hparam.MCDose_shape 
        doseGrid_shape = self.geometry.doseGrid.size.tolist()[::-1]
        dose = torch.zeros(doseGrid_shape, dtype=torch.float32, device=self.hparam.device)  # 3D dose @ doseGrid size
        dose[self.doseGrid_zz_yy_xx] = vector_dose  # index vector_dose to 3D dose
        dose = torch.nn.functional.interpolate(dose.view([1,1]+doseGrid_shape), size=(D//2,H,W), mode='trilinear', align_corners=False)  # interpolate only support 5D input
        #  dose = torch.nn.functional.interpolate(dose.view(1,1,D,H*2,W*2), size=(D//2,H,W), mode='nearest')
        dose = dose.squeeze()
        return dose

    def get_dose(self, uid):
        ''' return:pbDose (#slice, H, W) '''
        self._load_randApert()

        # get saved fluence of beam_id and aperture_id
        beam_id, apert_id = uid.split('_')
        beam_id, apert_id = int(beam_id), int(apert_id)
        preset_FM = self.dict_randomApertures[beam_id][apert_id]  # (H, W)
        dict_FMs = OrderedBunch()
        for bid, FM in self.data.dict_rayBoolMat.copy().items():
            if bid == beam_id:
                dict_FMs[bid] = FM * preset_FM.copy()  # FM may not all ones
            else:
                dict_FMs[bid] = FM*0
        fluence_vector = self.data.get_rays_from_fluences(dict_FMs)  
        dose = self.data.deposition.dot(fluence_vector)
        vector_dose = dose[0:self.data.get_pointNum_from_organName(self.roi_skinName)]  # only care the dose of skin
        pbDose = self._parse_dose_torch(torch.tensor(vector_dose, dtype=torch.float32)).cpu().numpy() # 3D dose 63x256x256
        pbDose = np.squeeze(pbDose).astype(np.float32)
        return pbDose

class MonteCarlo():
    def __init__(self, hparam, data):
        self.hparam = hparam
        self.data = data

        self.nb_leafPairs = 51    # 51 leaf pairs
        self.x_spacing    = 0.5   # cm
        self.nb_apertures = 1000 # we will generate this number random apertures
        self.nb_beams     = data.num_beams
        
        self._get_leafBottomEdgePosition()
        self._get_leafInJawField()  # get y axis leaf position from jaw_y1 ,jaw_y2

    def get_random_apertures(self):
        '''
        return: self.dict_randomApertures {beam_id: ndarray(nb_apertures, H, W)}
        '''
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

        # round to 2 decimals 
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

    def _get_x_axis_position(self):
        '''
         get x axis position from self.dict_randomApertures 
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

    def get_unit_MCdose(self):
        ''' Return: unitMUDose, ndarray (nb_beams*nb_apertures, #slice, H, W)  '''
        self._get_x_axis_position()  # get x axis position from the saved random generated fluences

        cprint(f'compute unit MU Dose on winServer and save results to {self.hparam.winServer_MonteCarloDir}', 'green')
        pdb.set_trace()
        if not Path(self.hparam.winServer_MonteCarloDir, 'Segs', 'Seg_6_999.txt').is_file(): 
            self.write_to_seg_txt()
        call_FM_gDPM_on_windowsServer(self.hparam.patient_ID, self.nb_beams, self.nb_apertures, hparam.winServer_nb_threads)
        pdb.set_trace()

    def get_dose(self, uid): 
        ''' return:mcDose(#slice, H, W) '''
        dpm_result_dir = Path(self.hparam.winServer_MonteCarloDir, 'gDPM_results', f'dpm_result_{uid}Ave.dat')
        with open(dpm_result_dir, 'rb') as f:
            dose = np.fromfile(f, dtype=np.float32)
            dose = dose.reshape(*hparam.MCDose_shape)
        mcDose = np.swapaxes(dose, 2, 1)
        return mcDose

def test_mcDose(beam_id, apert_id, npz_path):
    CTs = np.load(npz_path.joinpath('CTs.npz'))['CTs']  # TODO: CTs.shape != mcDose.shape
    # mcdose
    with open(f'/mnt/win_share/Chest_Pa26Plan12Rx14GPU/gDPM_results/dpm_result_{beam_id}_{apert_id}Ave.dat', 'rb') as f:
        dose = np.fromfile(f, dtype=np.float32)
        dose = dose.reshape((126,256,256))
        mcDose = np.swapaxes(dose, 2, 1)
    test_plot(CTs, mcDose, mcDose)
    print('done')

def test_mcDose_pbDose(uid, npz_path, is_rotation=False):
    beam_id, apert_id = uid.split('_')
    if is_rotation:
        CTs = np.load(npz_path.joinpath('CTs.npz'))['CTs'][int(beam_id)-1]
    else:
        CTs = np.load(npz_path.joinpath('CTs.npz'))['CTs']
    sample = np.load(npz_path.joinpath(f'mcpbDose_{beam_id}{apert_id.zfill(6)}.npz'))
    mcDose, pbDose = sample['mcDose'], sample['pbDose']
    test_plot(f'beamID{beam_id}_aptID{apert_id}', CTs, mcDose, pbDose)
    print(f'{uid} done')

def generate_mcDose_pbDose_dataset(data, mc, pb, npz_save_path):
    def resize_rotate_crop(dose, uid, margin=5):
        # resize
        dose = resize(dose, (126/2,256,256), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=True)
        dose = np.where(dose<0, 0, dose)  # bicubic(order=3) resize may create negative values

        # rotate
        angle = beam_info[int(uid.split('_')[0])].GantryAngle
        dose = np.moveaxis(dose, 0, -1)
        dose = rotate(dose, angle=angle, resize=False, center=pixel_isocenter, order=3, mode='constant', cval=0, clip=True, preserve_range=True)
        dose = np.moveaxis(dose, -1, 0)
        dose = np.where(dose<0, 0, dose)  # bicubic(order=3) resize may create negative values

        # crop: (256,256) -> (116, 177)
        if False:
            x1, x2 = pb.skin_lefttop.x-margin, pb.skin_rightbot.x+margin
            y1, y2 = pb.skin_lefttop.y-margin, pb.skin_rightbot.y+margin
            x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
            dose = dose[:, y1:y2, x1:x2]
        return dose.astype(np.float32)

    def process(uid):
        print(f'{uid}')
        mcDose = mc.get_dose(uid)
        pbDose = pb.get_dose(uid)
        print(f'mcDose max={mcDose.max()}')
        print(f'pbDose max={pbDose.max()}')
        mcDose = resize_rotate_crop(mcDose, uid)
        pbDose = resize_rotate_crop(pbDose, uid)
        #test_plot(CTs, mcDose, pbDose)
        assert pbDose.dtype == np.float32
        assert pbDose.min() >= 0
        assert pbDose.max() > 0
        assert mcDose.dtype == np.float32
        assert mcDose.min() >= 0
        assert mcDose.max() > 0
        assert mcDose.shape == pbDose.shape
        beam_id, apert_id = uid.split('_')
        save_path = npz_save_path.joinpath(f'mcpbDose_{beam_id}{apert_id.zfill(6)}.npz')
        npz_dict = {'mcDose':mcDose, 'pbDose':pbDose}
        np.savez(save_path, **npz_dict)

    def multiprocess(uids, nb_thread=10):
        for batch_uid in batch(uids, nb_thread):
            ps = []
            for uid in batch_uid:
                #process(uid)  # for test
                ps.append(Process(target=process, args=(uid,)))
                time.sleep(1)  # sleep 1s to avoid accessing winServer simultaneously 
                ps[-1].start()
            for p in ps:
                p.join()
    
    beam_info, pixel_isocenter = get_Dicom_info(data)

    # CT npz
    if not os.path.isfile(npz_save_path.joinpath('CTs.npz')):
        CTs = rescale_intensity(mc.data.Dicom_Reader.ArrayDicom, in_range='image', out_range=(0.0,1.0))  # TODO: in_range='image': min max of CTs; use HU range (-1024, 3071) instead?
        CTs = [resize_rotate_crop(CTs, f'{i}_0') for i in range(1, 6+1)]
        for ct in CTs:
            assert ct.dtype == np.float32
            assert ct.min() >= 0
        npz_dict = {'CTs': CTs}
        np.savez(npz_save_path.joinpath('CTs.npz'), **npz_dict)

    # doses npz
    uids = UIDs(npz_save_path).get_winServer_uids()
    multiprocess(uids)

def generate_mcDose_pbDose_dataset_Interp(hparam, data, mc, pb, npz_save_path):
    def center_crop(ndarray):
        tensor = torch.tensor(ndarray, dtype=torch.float32)
        tensor = torchvision.transforms.CenterCrop(128)(tensor)
        return tensor.cpu().numpy().astype(np.float32)

    def process(uid):
        #  print(f'{uid}')
        mcDose = mc.get_dose(uid) # 126x256x256
        mcDose = resize(mcDose, (D//2,H,W), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False)
        mcDose = np.where(mcDose<0, 0, mcDose).astype(np.float32)  # bicubic(order=3) resize may create negative values
        pbDose = pb.get_dose(uid) # 63x256x256
        print(f'{uid} mcDose max={mcDose.max()}')
        print(f'{uid} pbDose max={pbDose.max()}')
        mcDose = center_crop(mcDose)
        pbDose = center_crop(pbDose)
        #test_plot(CTs, mcDose, pbDose)
        assert pbDose.dtype == np.float32, 'pbDose not float32'
        assert pbDose.min() >= 0, f'pbDose.min {pbDose.min()}'
        assert pbDose.max() > 0,  f'pbDose.max {pbDose.max()}'
        assert mcDose.dtype == np.float32
        assert mcDose.min() >= 0, f'mcDose.min {mcDose.min()}'
        assert mcDose.max() > 0,  f'mcDose.max {mcDose.max()}'
        assert mcDose.shape == pbDose.shape, 'pbDose.shape != mcDose.shape'
        beam_id, apert_id = uid.split('_')
        save_path = npz_save_path.joinpath(f'mcpbDose_{beam_id}{apert_id.zfill(6)}.npz')
        npz_dict = {'mcDose':mcDose, 'pbDose':pbDose}
        np.savez(save_path, **npz_dict)
        print(f'saved {uid}')

    def multiprocess(uids, nb_thread=10):
        for batch_uid in batch(uids, nb_thread):
            print(f'processing: {batch_uid}')
            ps = []
            for uid in batch_uid:
                #  process(uid)  # for test
                #  pdb.set_trace()
                ps.append(Process(target=process, args=(uid,)))
                time.sleep(1)  # sleep 1s to avoid accessing winServer simultaneously 
                ps[-1].start()
            for p in ps:
                p.join()
    
    D, H, W = hparam.MCDose_shape 

    # CT npz
    if not os.path.isfile(npz_save_path.joinpath('CTs.npz')):
        CTs = rescale_intensity(mc.data.Dicom_Reader.ArrayDicom, in_range='image', out_range=(0.0,1.0))  # TODO: in_range='image': min max of CTs; use HU range (-1024, 3071) instead?
        CTs = resize(CTs, (D//2,H,W), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=True)
        CTs = np.where(CTs<0, 0, CTs)  # bicubic(order=3) resize may create negative values
        CTs = center_crop(CTs)
        assert CTs.dtype == np.float32
        assert CTs.min() >= 0
        npz_dict = {'CTs': CTs}
        np.savez(npz_save_path.joinpath('CTs.npz'), **npz_dict)
    # doses npz
    uids = UIDs(npz_save_path, Path(hparam.winServer_MonteCarloDir).joinpath('gDPM_results/dpm_result_*Ave.dat')).get_winServer_uids()
    multiprocess(uids)

def generate_mcDose_pbDose_dataset_npz_noRotation_noInterp(data, mc, pb, npz_save_path):
    def center_crop(ndarray):
        with torch.no_grad():
            tensor = torch.tensor(ndarray, dtype=torch.float32)
            tensor = torchvision.transforms.CenterCrop(128)(tensor)
            return tensor.cpu().numpy().astype(np.float32)

    def process(uid):
        #  print(f'{uid}')
        mcDose = mc.get_dose(uid) # 126x256x256
        D,H,W = hparam.MCDose_shape
        mcDose = resize(mcDose, (D/2,H,W), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=False)
        mcDose = np.where(mcDose<0, 0, mcDose).astype(np.float32)  # bicubic(order=3) resize may create negative values
        pbDose = pb.get_dose(uid, is_interp=False) # 63x256x256
        print(f'{uid} mcDose max={mcDose.max()}')
        print(f'{uid} pbDose max={pbDose.max()}')
        mcDose = center_crop(mcDose)
        print(f'{uid} crop mcdose')
        pbDose = center_crop(pbDose)
        print(f'{uid} crop pbdose')
        #test_plot(CTs, mcDose, pbDose)
        assert pbDose.dtype == np.float32, 'pbDose not float32'
        assert pbDose.min() >= 0, f'pbDose.min {pbDose.min()}'
        assert pbDose.max() > 0,  f'pbDose.max {pbDose.max()}'
        assert mcDose.dtype == np.float32
        assert mcDose.min() >= 0, f'mcDose.min {mcDose.min()}'
        assert mcDose.max() > 0,  f'mcDose.max {mcDose.max()}'
        assert mcDose.shape == pbDose.shape, 'pbDose.shape != mcDose.shape'
        beam_id, apert_id = uid.split('_')
        print(f'saving {uid}')
        save_path = npz_save_path.joinpath(f'mcpbDose_{beam_id}{apert_id.zfill(6)}.npz')
        npz_dict = {'mcDose':mcDose, 'pbDose':pbDose}
        np.savez(save_path, **npz_dict)
        print(f'saved {uid}')

    def multiprocess(uids, nb_thread=10):
        for batch_uid in batch(uids, nb_thread):
            print(batch_uid)
            ps = []
            for uid in batch_uid:
                # process(uid)  # for test
                ps.append(Process(target=process, args=(uid,)))
                time.sleep(1)  # sleep 1s to avoid accessing winServer simultaneously 
                ps[-1].start()
            for p in ps:
                p.join()
    
    beam_info, pixel_isocenter = get_Dicom_info(data)

    # CT npz
    if not os.path.isfile(npz_save_path.joinpath('CTs.npz')):
        CTs = rescale_intensity(mc.data.Dicom_Reader.ArrayDicom, in_range='image', out_range=(0.0,1.0))  # TODO: in_range='image': min max of CTs; use HU range (-1024, 3071) instead?
        CTs = resize(CTs, (126//2,256,256), order=3, mode='constant', cval=0, clip=False, preserve_range=True, anti_aliasing=True)
        CTs = np.where(CTs<0, 0, CTs)  # bicubic(order=3) resize may create negative values
        CTs = center_crop(CTs)
        assert CTs.dtype == np.float32
        assert CTs.min() >= 0
        npz_dict = {'CTs': CTs}
        np.savez(npz_save_path.joinpath('CTs.npz'), **npz_dict)
    # doses npz
    uids = UIDs(npz_save_path).get_winServer_uids()
    multiprocess(uids)

def generate_h5Files(hparam):
    data_path = Path('/mnt/ssd/tps_optimization/').joinpath(hparam.patient_ID).joinpath('pbmcDoses_npz_rotation')
    train_list = list(braceexpand(str(data_path.joinpath('mcpbDose_{1..6}{000000..000989}.npz'))))
    valid_list = list(braceexpand(str(data_path.joinpath('mcpbDose_{1..6}{000990..000999}.npz'))))

    # with h5py.File(npz_path.joinpath('all_pbmcDoses.h5'), 'w') as h5f:
    with h5py.File('./all_pbmcDoses.h5', 'w') as h5f:
        ds_train = h5f.create_dataset('pb_mc_doses_train', (len(train_list), 2, 63, 128, 128), dtype=np.float32)
        ds_uid_train = h5f.create_dataset('UIDs_train',    (len(train_list), ), dtype='S16')
        ds_valid = h5f.create_dataset('pb_mc_doses_valid', (len(valid_list), 2, 63, 128, 128), dtype=np.float32)
        ds_uid_valid = h5f.create_dataset('UIDs_valid',    (len(valid_list), ), dtype='S16')

        for i, fn in enumerate(train_list):
            uid = str.encode(os.path.basename(fn))  # encode string to bytes
            print(uid)
            pbmcDose = np.load(fn)
            ds_train[i, 0] = pbmcDose['pbDose'][:, 64:64+128, 64:64+128]  # center crop
            ds_train[i, 1] = pbmcDose['mcDose'][:, 64:64+128, 64:64+128]
            ds_uid_train[i] = uid 
        for i, fn in enumerate(valid_list):
            uid = str.encode(os.path.basename(fn))  # encode string to bytes
            print(uid)
            pbmcDose = np.load(fn)
            ds_valid[i, 0] = pbmcDose['pbDose'][:, 64:64+128, 64:64+128]  # center crop
            ds_valid[i, 1] = pbmcDose['mcDose'][:, 64:64+128, 64:64+128]
            ds_uid_valid[i] = uid 
        h5f.flush()
    cprint('convert to h5 done', 'green')

def main(hparam):
    #  npz_path = Path('/mnt/ssd/tps_optimization/').joinpath(hparam.patient_ID).joinpath('pbmcDoses_npz_noRotation_noInterp')
    npz_path = Path('/mnt/ssd/tps_optimization/').joinpath(hparam.patient_ID).joinpath('pbmcDoses_npz_Interp')
    make_dir(npz_path)

    if hparam.test_mcDose:
        test_mcDose(4, 692, npz_path)

    if hparam.test_pbmcDoses:
        for beam_id in range(1, 2):
            test_mcDose_pbDose(f'{beam_id}_1', npz_path)
            test_mcDose_pbDose(f'{beam_id}_2', npz_path)
            test_mcDose_pbDose(f'{beam_id}_3', npz_path)

    if hparam.npz2h5:
        generate_h5Files(hparam)

    data = Data(hparam)
    mc = MonteCarlo(hparam, data)

    if hparam.Calculate_MC_unit_doses:
        mc.get_random_apertures()
        mc.get_unit_MCdose()

    pb = PencilBeam(hparam, data)

    if hparam.mcpbDose2npz:
        pb = PencilBeam(hparam, data)
        generate_mcDose_pbDose_dataset(data, mc, pb, npz_path)

    if hparam.mcpbDose2npz_noRotation_noInterp:
        generate_mcDose_pbDose_dataset_npz_noRotation_noInterp(data, mc, pb, npz_path)

    if hparam.mcpbDose2npz_Interp:
        generate_mcDose_pbDose_dataset_Interp(hparam, data, mc, pb, npz_path)

if __name__ == "__main__":
    hparam = BaseOptions().parse()
    main(hparam)
