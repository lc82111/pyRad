import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os, copy, pydicom, math, glob
from orderedbunch import OrderedBunch
from scipy.ndimage import median_filter 
import numpy.random as npr
from termcolor import cprint
from pydicom.tag import Tag
import SimpleITK as sitk
from skimage import draw
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label,regionprops,find_contours

from argparse import ArgumentParser
from io import StringIO
from shutil import copyfile
import sys, collections, shutil, pdb, pickle, datetime

import torch

def get_now_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pickle_object(title, data):
    pikd = open(title, 'wb')
    pickle.dump(data, pikd, protocol=4)
    pikd.close()

def unpickle_object(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

def to_np(x):
    return x.detach().cpu().numpy()

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        cprint(f'making new dir {path}', 'yellow')

def cp(src, dst):
    if not os.path.isfile(dst):
        copyfile(src, dst)

def del_fold(dir_path):
    try:
        shutil.rmtree(dir_path, ignore_errors=True)
    except OSError as e:
        print("Warning: %s : %s" % (dir_path, e.strerror))

def split_doses(doses, organ_inf):
    #assert to_np(torch.isnan(doses)).any() == False
    list_organ_doses = torch.split(doses, list(organ_inf.values()))
    dict_organ_doses = collections.OrderedDict()

    for organ_name, organ_dose in zip(list(organ_inf.keys()), list_organ_doses):
        dict_organ_doses[organ_name] = organ_dose

    return dict_organ_doses 

def parse_MonteCarlo_dose(MCDose, data):
    ''' Return: dict_organ_dose {organ_name: dose ndarray (#organ_dose, )} '''
    dict_organ_dose = OrderedBunch()
    for organ_name, msk in data.organ_masks.items():
        assert MCDose.shape == msk.shape 
        dict_organ_dose[organ_name] = MCDose[msk]
    return dict_organ_dose 

def call_FM_gDPM_on_windowsServer(PID, nb_beams, nb_apertures, nb_threads, host_ip="192.168.10.103", port=13000):
    cprint(f'send msg to windows Server to call FM.exe and gDPM.exe.', 'green')

    addr = (host_ip, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    msg = "seg.txt files are ready;%s;%s;%s;%s"%(PID, nb_beams, nb_apertures, nb_threads)
    data = msg.encode('utf-8')
    UDPSock.sendto(data, addr)
    UDPSock.close()
    cprint("messages send.", 'green')

    try:
        host_ip = "0.0.0.0"
        port = 13001
        buf = 1024
        addr = (host_ip, port)
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

def get_segment_grad(dict_segments, dict_rayBoolMat):
    dict_gradMaps = OrderedBunch()
    for beam_id, mask in dict_rayBoolMat.items(): # for each beam
        grad = dict_segments[beam_id].grad.detach().cpu().numpy()  # (h*w, #aperture=1)
        grad = grad.sum(axis=-1)  # (h*w)
        dict_gradMaps[beam_id] = grad.reshape(*mask.shape) * mask
    return dict_gradMaps

def multiply_dict(dict_gradMaps, dict_segments):
    ''' dot product grad and seg
    Arguments:
        grads: {beam_id: matrix}
        dict_segments: {beam_id: vector}
    Return: 
        ret: scalar '''
    ret = 0
    for segment, gradMap in zip(dict_segments.values(), dict_gradMaps.values()):
        ret += np.dot(gradMap.flatten(), segment)
    return ret

def smallest_contiguous_sum(array):
    '''
    array: 1D vector
    return: 1D vector
    '''
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
    return segment

def _test_smallest_contiguous_sum():
    #  test_array = np.random.rand((10,), dtype=np.float32) * 2 - 1.
    test_array = np.array([1,1,1,99,99,99, -1, -3, -1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

    test_array = np.array([-1,-1,-1,99,99,99, -1, -3, -1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

    test_array = np.array([-1,-1,-1,99,-10,99, -1, -3, -1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

    test_array = np.array([-1,-1,-1, -1, -3, -1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

    test_array = np.array([1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

    test_array = np.array([-1])
    print(test_array)
    seg = smallest_contiguous_sum(test_array)
    cprint(seg.astype(np.uint8), 'red')

def computer_fluence(data, dict_segments, dict_MUs):
    '''computer fluence from seg and MU. 
    data: a data class instance
    dict_segments: {beam_id: matrix consists of segment columns}
    dict_MUs:{beam_id: vector of segment MU}
    return: fluence (#valid_bixels, )
    For retain_grad() see : https://discuss.pytorch.org/t/how-do-i-calculate-the-gradients-of-a-non-leaf-variable-w-r-t-to-a-loss-function/5112
    '''
    dict_fluences = OrderedBunch()
    for beam_id, seg in dict_segments.items():
        MU = torch.abs(dict_MUs[beam_id]) # nonnegative constraint 
        dict_fluences[beam_id] = torch.matmul(seg, MU)  # {beam_id: vector}

    fluence, dict_fluenceMaps = data.project_to_validRays_torch(dict_fluences)  # (#valid_bixels,), {beam_id: matrix}
    fluence.retain_grad()
    return fluence, dict_fluenceMaps

def cal_dose(D, fluence):
    ''' cal dose from depos and fluence
    Arguments:
        fluence: tensor (#bixel, )
        D deposition: tensor (#dose_points, #bixel)
    Return: dose: tensor (#dose_points, )
    '''
    if fluence.dim() == 1:
        fluence = fluence.unsqueeze(dim=1)
    if D.is_sparse:  # sparse matrix
        dose = torch.sparse.mm(D, fluence) 
    else:
        dose = torch.mm(D, fluence) 
    return dose.squeeze()

def convert_depoMatrix_to_tensor(D, device):
    """ Convert a scipy Sparse Matrix D to a torch Sparse Tensor when necessary"""
    # D matrix (#voxels, #bixels)
    if type(D).__module__ == 'scipy.sparse.coo':  # sparse matrix
        D = torch.sparse.FloatTensor(torch.LongTensor([D.row, D.col]), torch.tensor(D.data), torch.Size(D.shape))
        D = D.to(device=device, dtype=torch.float32)
    elif isinstance(D, OrderedBunch):
        for beam_id, d in D.items():
            #  d = torch.sparse.FloatTensor(torch.LongTensor([d.row, d.col]), torch.tensor(d.data), torch.Size(d.shape))
            d = torch.sparse_coo_tensor(torch.LongTensor([d.row, d.col]), torch.tensor(d.data), torch.Size(d.shape), dtype=torch.float32, device=device)
            D.update({beam_id: d})
    else: # dense matrix
        D = torch.tensor(D, dtype=torch.float32, device=self.hparam.device)
    return D


def load_npz(npz_path):
    # npz object can not be access in parallel (i.e., num_workers>0 in torch.dataloader), therefore we setup a new dict to enable parallel.
    new_npz = {}
    data_npz = np.load(npz_path)
    for k, v in data_npz.items():
        new_npz[k] = v
    return new_npz

def add_colorbar(fig, axe, img):
    divider = make_axes_locatable(axe)
    ax_cb = divider.new_vertical(size="10%", pad=0.3, pack_start=True)
    fig.add_axes(ax_cb)
    fig.colorbar(img, cax=ax_cb,  orientation="horizontal")

class Dicom_to_Imagestack:
    def __init__(self, rewrite_RT_file=False, delete_previous_rois=True,Contour_Names=None,
                 template_dir=None, channels=3, get_images_mask=True, arg_max=True,
                 associations={},desc='',iteration=0, get_dose_output=False, **kwargs):
        self.get_dose_output = get_dose_output
        self.associations = associations
        self.set_contour_names(Contour_Names)
        self.set_associations(associations)
        self.set_get_images_and_mask(get_images_mask)
        self.set_description(desc)
        self.set_iteration(iteration)
        self.arg_max = arg_max
        self.rewrite_RT_file = rewrite_RT_file
        self.dose_handles = []
        if template_dir is None:
            package_name = __package__.split('.')[-1]
            template_dir = os.path.join(__file__[:__file__.index(package_name)],package_name,'template_RS.dcm')
        self.template_dir = template_dir
        self.template = True
        self.delete_previous_rois = delete_previous_rois
        self.channels = channels
        self.get_images_mask = get_images_mask
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.__reset__()

    def __reset__(self):
        self.all_RTs = {}
        self.all_rois = []
        self.all_paths = []
        self.paths_with_contours = []

    def set_associations(self, associations={}):
        keys = list(associations.keys())
        for key in keys:
            associations[key.lower()] = associations[key].lower()
        if self.Contour_Names is not None:
            for name in self.Contour_Names:
                if name not in associations:
                    associations[name] = name
        self.associations, self.hierarchy = associations, {}

    def set_get_images_and_mask(self, get_images_mask=True):
        self.get_images_mask = get_images_mask

    def set_contour_names(self, Contour_Names=None):
        self.__reset__()
        if Contour_Names is None:
            Contour_Names = []
        else:
            Contour_Names = [i.lower() for i in Contour_Names]
        self.Contour_Names = Contour_Names
        self.set_associations(self.associations)

    def set_description(self, description):
        self.desciption = description

    def set_iteration(self, iteration=0):
        self.iteration = str(iteration)

    def down_folder(self, input_path, reset=True):
        files = []
        dirs = []
        file = []
        for root, dirs, files in os.walk(input_path):
            break
        for val in files:
            if val.find('.dcm') != -1:
                file = val
                break
        if file and input_path:
            self.all_paths.append(input_path)
            self.Make_Contour_From_directory(input_path)
        for dir in dirs:
            new_directory = os.path.join(input_path, dir)
            self.down_folder(new_directory)
        return None

    def make_array(self, PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = None
        self.Dicom_info = []
        fileList = []
        self.RTs_in_case = {}
        self.RDs_in_case = {}
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        if not self.get_images_mask:
            RT_fileList = [i for i in fileList if i.find('RT') == 0 or i.find('RS') == 0]
            if RT_fileList:
                fileList = RT_fileList
            for filename in fileList:
                try:
                    ds = pydicom.read_file(os.path.join(dirName, filename), force=True)
                    self.ds = ds
                    if ds.Modality == 'CT' or ds.Modality == 'MR' or ds.Modality == 'PT':
                        self.lstFilesDCM.append(os.path.join(dirName, filename))
                        self.Dicom_info.append(ds)
                        self.ds = ds
                    elif ds.Modality == 'RTSTRUCT':
                        self.lstRSFile = os.path.join(dirName, filename)
                        self.RTs_in_case[self.lstRSFile] = []
                except:
                    #continue
                    raise ValueError
            if self.lstFilesDCM:
                self.RefDs = pydicom.read_file(self.lstFilesDCM[0])
        else:
            self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
            self.reader.SetFileNames(self.dicom_names)
            self.get_images()
            image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
            RT_Files = [os.path.join(PathDicom, file) for file in fileList if file not in image_files]
            reader = sitk.ImageFileReader()
            for lstRSFile in RT_Files:
                reader.SetFileName(lstRSFile)
                try:
                    reader.ReadImageInformation()
                    modality = reader.GetMetaData("0008|0060")
                except:
                    modality = pydicom.read_file(lstRSFile, force=True).Modality
                if modality.lower().find('dose') != -1:
                    self.RDs_in_case[lstRSFile] = []
                elif modality.lower().find('struct') != -1:
                    self.RTs_in_case[lstRSFile] = []
                elif modality.lower().find('rtplan') != -1:
                    self.set_isocenter_and_beam_angle(lstRSFile)
            self.RefDs = pydicom.read_file(self.dicom_names[0])
            self.ds = pydicom.read_file(self.dicom_names[0])
        self.all_contours_exist = False
        self.rois_in_case = []
        self.all_RTs.update(self.RTs_in_case)
        if len(self.RTs_in_case.keys()) > 0:
            self.template = False
            for self.lstRSFile in self.RTs_in_case:
                self.get_rois_from_RT()
        elif self.get_images_mask:
            self.use_template()

    def set_isocenter_and_beam_angle(self, rtplan_file):
        ds = pydicom.read_file(rtplan_file, force=True)
        self.beam_info = OrderedBunch()
        for i, beam in enumerate(ds.BeamSequence):
            self.beam_info[i+1] = OrderedBunch()
            cp0 = beam.ControlPointSequence[0]
            self.beam_info[i+1].SSD = float(cp0.SourceToSurfaceDistance / 10)
            self.beam_info[i+1].GantryAngle = float(cp0.GantryAngle)
            self.beam_info[i+1].IsoCenter = np.array([float(x) for x in cp0.IsocenterPosition])

    def get_rois_from_RT(self):
        rois_in_structure = []
        self.RS_struct = pydicom.read_file(self.lstRSFile, force=True)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        for Structures in self.ROI_Structure:
            if Structures.ROIName not in self.rois_in_case:
                self.rois_in_case.append(Structures.ROIName)
                rois_in_structure.append(Structures.ROIName)
        self.all_RTs[self.lstRSFile] = rois_in_structure

    def get_mask(self):
        self.mask = np.zeros([len(self.dicom_names), self.image_size_rows, self.image_size_cols, len(self.Contour_Names) + 1],
                             dtype='int8')
        self.structure_references = {}
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[
                self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
        found_rois = {}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name].lower()
            elif ROI_Name.lower() in self.associations:
                true_name = self.associations[ROI_Name.lower()]
            if true_name and true_name in self.Contour_Names:
                found_rois[true_name] = {'Hierarchy': 999, 'Name': ROI_Name, 'Roi_Number': Structures.ROINumber}
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.Contours_to_mask(index)
                self.mask[..., self.Contour_Names.index(ROI_Name) + 1][mask == 1] = 1
        if self.arg_max:
            self.mask = np.argmax(self.mask, axis=-1)
        self.annotation_handle = sitk.GetImageFromArray(self.mask.astype('int8'))
        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        return None

    def Contours_to_mask(self, i):
        mask = np.zeros([len(self.dicom_names), self.image_size_rows, self.image_size_cols], dtype='int8')
        Contour_data = self.RS_struct.ROIContourSequence[i].ContourSequence
        shifts = [[float(i) for i in self.reader.GetMetaData(j, "0020|0032").split('\\')] for j in range(len(self.reader.GetFileNames()))]
        Xx, Xy, Xz, Yx, Yy, Yz = [float(i) for i in self.reader.GetMetaData(0, "0020|0037").split('\\')]
        PixelSize = self.dicom_handle.GetSpacing()
        mult1 = mult2 = 1

        for i in range(len(Contour_data)):
            referenced_sop_instance_uid = Contour_data[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            if referenced_sop_instance_uid not in self.SOPInstanceUIDs:
                print('{} Error here with instance UID'.format(self.PathDicom))
                return None
            else:
                slice_index = self.SOPInstanceUIDs.index(referenced_sop_instance_uid)
            ShiftRowsBase, ShiftColsBase, ShiftzBase = shifts[slice_index]
            if ShiftRowsBase > 0:
                mult1 = -1
            ShiftRows = ShiftRowsBase * Xx + ShiftColsBase * Xy + ShiftzBase * Xz
            ShiftCols = ShiftRowsBase * Xy + ShiftColsBase * Yy + ShiftzBase * Yz

            rows = Contour_data[i].ContourData[1::3]
            cols = Contour_data[i].ContourData[0::3]
            row_val = [abs(x - mult1 * ShiftRows)/PixelSize[0] for x in cols]
            col_val = [abs(x - mult2 * ShiftCols)/PixelSize[1] for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.image_size_rows, self.image_size_cols])
            mask[slice_index, :, :][temp_mask > 0] += 1
        mask = mask % 2
        return mask

    def use_template(self):
        self.template = True
        if not self.template_dir:
            self.template_dir = os.path.join('\\\\mymdafiles', 'ro-admin', 'SHARED', 'Radiation physics', 'BMAnderson',
                                             'Auto_Contour_Sites', 'template_RS.dcm')
            if not os.path.exists(self.template_dir):
                self.template_dir = os.path.join('..', '..', 'Shared_Drive', 'Auto_Contour_Sites', 'template_RS.dcm')
        self.key_list = self.template_dir.replace('template_RS.dcm', 'key_list.txt')
        self.RS_struct = pydicom.read_file(self.template_dir)
        print('Running off a template')
        self.changetemplate()

    def get_images(self):
        self.dicom_handle = self.reader.Execute()
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i, sop_instance_UID_key) for i in
                                range(self.dicom_handle.GetDepth())]
        slice_location_key = "0020|0032"
        self.slice_info = [self.reader.GetMetaData(i, slice_location_key).split('\\')[-1] for i in
                           range(self.dicom_handle.GetDepth())]

        #  self.CT_imagePositions = [self.reader.GetMetaData(i, '0020|0032').split('\\') for i in range(self.dicom_handle.GetDepth())]
        #  self.CT_spacing = [self.reader.GetMetaData(i, '0028|0030').split('\\') for i in range(self.dicom_handle.GetDepth())]

        self.ArrayDicom = sitk.GetArrayFromImage(self.dicom_handle)
        self.image_size_cols, self.image_size_rows, self.image_size_z = self.dicom_handle.GetSize()

    def write_images_annotations(self, out_path):
        image_path = os.path.join(out_path, 'Overall_Data_{}_{}.nii.gz'.format(self.desciption, self.iteration))
        annotation_path = os.path.join(out_path, 'Overall_mask_{}_y{}.nii.gz'.format(self.desciption,self.iteration))
        if os.path.exists(image_path):
            return None
        pixel_id = self.dicom_handle.GetPixelIDTypeAsString()
        if pixel_id.find('32-bit signed integer') != 0:
            self.dicom_handle = sitk.Cast(self.dicom_handle, sitk.sitkFloat32)
        sitk.WriteImage(self.dicom_handle,image_path)

        self.annotation_handle.SetSpacing(self.dicom_handle.GetSpacing())
        self.annotation_handle.SetOrigin(self.dicom_handle.GetOrigin())
        self.annotation_handle.SetDirection(self.dicom_handle.GetDirection())
        pixel_id = self.annotation_handle.GetPixelIDTypeAsString()
        if pixel_id.find('int') == -1:
            self.annotation_handle = sitk.Cast(self.annotation_handle, sitk.sitkUInt8)
        sitk.WriteImage(self.annotation_handle,annotation_path)
        if len(self.dose_handles) > 0:
            for dose_index, dose_handle in enumerate(self.dose_handles):
                if len(self.dose_handles) > 1:
                    dose_path = os.path.join(out_path,
                                             'Overall_dose_{}_{}_{}.nii.gz'.format(self.desciption, self.iteration,
                                                                                   dose_index))
                else:
                    dose_path = os.path.join(out_path,
                                             'Overall_dose_{}_{}.nii.gz'.format(self.desciption, self.iteration))
                sitk.WriteImage(dose_handle, dose_path)
        fid = open(os.path.join(self.PathDicom, self.desciption + '_Iteration_' + self.iteration + '.txt'), 'w+')
        fid.close()

    def poly2mask(self, vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

    def with_annotations(self, annotations, output_dir, ROI_Names=None):
        assert ROI_Names is not None, 'You need to provide ROI_Names'
        annotations = np.squeeze(annotations)
        self.image_size_z, self.image_size_rows, self.image_size_cols = annotations.shape[:3]
        self.ROI_Names = ROI_Names
        self.output_dir = output_dir
        if len(annotations.shape) == 3:
            annotations = np.expand_dims(annotations, axis=-1)
        self.annotations = annotations
        self.Mask_to_Contours()

    def Mask_to_Contours(self):
        self.RefDs = self.ds
        self.shift_list = [[float(i) for i in self.reader.GetMetaData(j, "0020|0032").split('\\')] for j in range(len(self.reader.GetFileNames()))] #ShiftRows, ShiftCols, ShiftZBase
        self.mv = Xx, Xy, Xz, Yx, Yy, Yz = [float(i) for i in self.reader.GetMetaData(0, "0020|0037").split('\\')]
        self.ShiftRows = [i[0] * Xx + i[1] * Xy + i[2] * Xz for i in self.shift_list]
        self.ShiftCols = [i[0] * Xy + i[1] * Yy + i[2] * Yz for i in self.shift_list]
        self.ShiftZ = [i[2] for i in self.shift_list]
        self.mult1 = self.mult2 = 1
        self.PixelSize = self.dicom_handle.GetSpacing()
        current_names = []
        for names in self.RS_struct.StructureSetROISequence:
            current_names.append(names.ROIName)
        Contour_Key = {}
        xxx = 1
        for name in self.ROI_Names:
            Contour_Key[name] = xxx
            xxx += 1
        self.all_annotations = self.annotations
        base_annotations = copy.deepcopy(self.annotations)
        temp_color_list = []
        color_list = [[128, 0, 0], [170, 110, 40], [0, 128, 128], [0, 0, 128], [230, 25, 75], [225, 225, 25],
                      [0, 130, 200], [145, 30, 180],
                      [255, 255, 255]]
        self.struct_index = 0
        new_ROINumber = 1000
        for Name in self.ROI_Names:
            new_ROINumber -= 1
            if not temp_color_list:
                temp_color_list = copy.deepcopy(color_list)
            color_int = np.random.randint(len(temp_color_list))
            print('Writing data for ' + Name)
            self.annotations = copy.deepcopy(base_annotations[:, :, :, int(self.ROI_Names.index(Name) + 1)])
            self.annotations = self.annotations.astype('int')

            make_new = 1
            allow_slip_in = True
            if (Name not in current_names and allow_slip_in) or self.delete_previous_rois:
                self.RS_struct.StructureSetROISequence.insert(0,copy.deepcopy(self.RS_struct.StructureSetROISequence[0]))
            else:
                print('Prediction ROI {} is already within RT structure'.format(Name))
                continue
            self.RS_struct.StructureSetROISequence[self.struct_index].ROINumber = new_ROINumber
            self.RS_struct.StructureSetROISequence[self.struct_index].ReferencedFrameOfReferenceUID = \
                self.ds.FrameOfReferenceUID
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIName = Name
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIVolume = 0
            self.RS_struct.StructureSetROISequence[self.struct_index].ROIGenerationAlgorithm = 'SEMIAUTOMATIC'
            if make_new == 1:
                self.RS_struct.RTROIObservationsSequence.insert(0,
                    copy.deepcopy(self.RS_struct.RTROIObservationsSequence[0]))
                if 'MaterialID' in self.RS_struct.RTROIObservationsSequence[self.struct_index]:
                    del self.RS_struct.RTROIObservationsSequence[self.struct_index].MaterialID
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ObservationNumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            self.RS_struct.RTROIObservationsSequence[self.struct_index].ROIObservationLabel = Name
            self.RS_struct.RTROIObservationsSequence[self.struct_index].RTROIInterpretedType = 'ORGAN'

            if make_new == 1:
                self.RS_struct.ROIContourSequence.insert(0,copy.deepcopy(self.RS_struct.ROIContourSequence[0]))
            self.RS_struct.ROIContourSequence[self.struct_index].ReferencedROINumber = new_ROINumber
            del self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[1:]
            self.RS_struct.ROIContourSequence[self.struct_index].ROIDisplayColor = temp_color_list[color_int]
            del temp_color_list[color_int]
            thread_count = int(cpu_count()*0.9-1)
            # thread_count = 1
            contour_dict = {}
            q = Queue(maxsize=thread_count)
            threads = []
            kwargs = {'image_size_rows': self.image_size_rows, 'image_size_cols': self.image_size_cols,
                      'slice_info': self.slice_info, 'PixelSize': self.PixelSize, 'mult1': self.mult1,
                      'mult2': self.mult2, 'ShiftZ': self.ShiftZ, 'mv': self.mv, 'shift_list': self.shift_list,
                      'ShiftRows': self.ShiftRows, 'ShiftCols': self.ShiftCols, 'contour_dict': contour_dict}

            A = [q,kwargs]
            for worker in range(thread_count):
                t = Thread(target=contour_worker, args=(A,))
                t.start()
                threads.append(t)

            contour_num = 0
            if np.max(self.annotations) > 0:  # If we have an annotation, write it
                image_locations = np.max(self.annotations, axis=(1, 2))
                indexes = np.where(image_locations > 0)[0]
                for index in indexes:
                    item = [self.annotations[index, ...], index]
                    q.put(item)
                for i in range(thread_count):
                    q.put(None)
                for t in threads:
                    t.join()
                for i in contour_dict.keys():
                    for output in contour_dict[i]:
                        if contour_num > 0:
                            self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence.append(
                                copy.deepcopy(
                                    self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[0]))
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourNumber = str(contour_num)
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourImageSequence[0].ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].ContourData = output
                        self.RS_struct.ROIContourSequence[self.struct_index].ContourSequence[
                            contour_num].NumberofContourPoints = round(len(output) / 3)
                        contour_num += 1
        self.RS_struct.SOPInstanceUID += '.' + str(np.random.randint(999))
        if self.template or self.delete_previous_rois:
            for i in range(len(self.RS_struct.StructureSetROISequence),len(self.ROI_Names),-1):
                del self.RS_struct.StructureSetROISequence[-1]
            for i in range(len(self.RS_struct.RTROIObservationsSequence),len(self.ROI_Names),-1):
                del self.RS_struct.RTROIObservationsSequence[-1]
            for i in range(len(self.RS_struct.ROIContourSequence),len(self.ROI_Names),-1):
                del self.RS_struct.ROIContourSequence[-1]
            for i in range(len(self.RS_struct.StructureSetROISequence)):
                self.RS_struct.StructureSetROISequence[i].ROINumber = i + 1
                self.RS_struct.RTROIObservationsSequence[i].ReferencedROINumber = i + 1
                self.RS_struct.ROIContourSequence[i].ReferencedROINumber = i + 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        out_name = os.path.join(self.output_dir,
                                'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '.dcm')
        if os.path.exists(out_name):
            out_name = os.path.join(self.output_dir,
                                    'RS_MRN' + self.RS_struct.PatientID + '_' + self.RS_struct.SeriesInstanceUID + '1.dcm')
        print('Writing out data...')
        pydicom.write_file(out_name, self.RS_struct)
        fid = open(os.path.join(self.output_dir, 'Completed.txt'), 'w+')
        fid.close()
        print('Finished!')
        return None

    def changetemplate(self):
        keys = self.RS_struct.keys()
        for key in keys:
            # print(self.RS_struct[key].name)
            if self.RS_struct[key].name == 'Referenced Frame of Reference Sequence':
                break
        self.RS_struct[key]._value[0].FrameOfReferenceUID = self.ds.FrameOfReferenceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].ReferencedSOPInstanceUID = self.ds.StudyInstanceUID
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
            0].SeriesInstanceUID = self.ds.SeriesInstanceUID
        for i in range(len(self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                               0].ContourImageSequence) - 1):
            del self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[-1]
        fill_segment = copy.deepcopy(
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence[0])
        for i in range(len(self.SOPInstanceUIDs)):
            temp_segment = copy.deepcopy(fill_segment)
            temp_segment.ReferencedSOPInstanceUID = self.SOPInstanceUIDs[i]
            self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[
                0].ContourImageSequence.append(temp_segment)
        del \
        self.RS_struct[key]._value[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence[0]

        new_keys = open(self.key_list)
        keys = {}
        i = 0
        for line in new_keys:
            keys[i] = line.strip('\n').split(',')
            i += 1
        new_keys.close()
        for index in keys.keys():
            new_key = keys[index]
            try:
                self.RS_struct[new_key[0], new_key[1]] = self.ds[[new_key[0], new_key[1]]]
            except:
                continue
        return None

    def get_dose(self):
        reader = sitk.ImageFileReader()
        output, spacing, direction, origin = None, None, None, None
        for dose_file in self.RDs_in_case:
            if os.path.split(dose_file)[-1].startswith('RTDOSE - PLAN'):
                reader.SetFileName(dose_file)
                reader.ReadImageInformation()
                dose = reader.Execute()
                spacing = dose.GetSpacing()
                origin = dose.GetOrigin()
                direction = dose.GetDirection()
                scaling_factor = float(reader.GetMetaData("3004|000e"))
                dose = sitk.GetArrayFromImage(dose)*scaling_factor
                if output is None:
                    output = dose
                else:
                    output += dose
        if output is not None:
            output = sitk.GetImageFromArray(output)
            output.SetSpacing(spacing)
            output.SetDirection(direction)
            output.SetOrigin(origin)
            self.dose_handles.append(output)

    def Make_Contour_From_directory(self, PathDicom):
        self.make_array(PathDicom)
        if self.rewrite_RT_file:
            self.rewrite_RT()
        if not self.template and self.get_images_mask:
            self.get_mask()
        if self.get_dose_output:
            self.get_dose()
        true_rois = []
        for roi in self.rois_in_case:
            if roi not in self.all_rois:
                self.all_rois.append(roi)
            if self.Contour_Names:
                if roi.lower() in self.associations:
                    true_rois.append(self.associations[roi.lower()])
                elif roi.lower() in self.Contour_Names:
                    true_rois.append(roi.lower())
        self.all_contours_exist = True
        for roi in self.Contour_Names:
            if roi not in true_rois:
                print('Lacking {} in {}'.format(roi, PathDicom))
                print('Found {}'.format(self.rois_in_case))
                self.all_contours_exist = False
                break
        if PathDicom not in self.paths_with_contours and self.all_contours_exist:
            self.paths_with_contours.append(PathDicom) # Add the path that has the contours
        return None

    def rewrite_RT(self, lstRSFile=None):
        if lstRSFile is not None:
            self.RS_struct = pydicom.read_file(lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        if Tag((0x3006, 0x080)) in self.RS_struct.keys():
            self.Observation_Sequence = self.RS_struct.RTROIObservationsSequence
        else:
            self.Observation_Sequence = []
        self.rois_in_case = []
        for i, Structures in enumerate(self.ROI_Structure):
            if Structures.ROIName in self.associations:
                new_name = self.associations[Structures.ROIName]
                self.RS_struct.StructureSetROISequence[i].ROIName = new_name
            self.rois_in_case.append(self.RS_struct.StructureSetROISequence[i].ROIName)
        for i, ObsSequence in enumerate(self.Observation_Sequence):
            if ObsSequence.ROIObservationLabel in self.associations:
                new_name = self.associations[ObsSequence.ROIObservationLabel]
                self.RS_struct.RTROIObservationsSequence[i].ROIObservationLabel = new_name
        self.RS_struct.save_as(self.lstRSFile)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx+n, l)]

class UIDs():
    def __init__(self, npz_path, winServer_gDPM_result_path):
        self.npz_fns = [os.path.basename(fn) for fn in glob.glob(str(npz_path.joinpath("mcpbDose_*.npz")))]
        self.winServer_gDPM_result_path = winServer_gDPM_result_path
    def in_npz_uids(self, uid):
        beam_id, apert_id = uid.split('_')
        return f'mcpbDose_{beam_id}{apert_id.zfill(6)}.npz' in self.npz_fns
    def get_winServer_uids(self):
        dpm_results_fns = [os.path.basename(fn) for fn in glob.glob(str(self.winServer_gDPM_result_path))]
        uids = []
        for fn in dpm_results_fns:  
            uid = fn.strip('dpm_results_').strip('Ave.data')
            if not self.in_npz_uids(uid): uids.append(uid)
        print(f'following uids will be convert {uids}')
        return uids

def test_plot(uid, CTs, mcDose, pbDose):
    time = get_now_time()
    make_dir(f'test/{time}')
    for i, (ct, md, pd) in enumerate(zip(CTs, mcDose, pbDose)): 
        figs, axes = plt.subplots(1, 2, figsize=(2*10, 1*10))
        figs.set_tight_layout(True)

        axes[0].imshow(ct, cmap='gray')
        _ = axes[0].imshow(md, cmap='jet', alpha=0.6)
        add_colorbar(figs, axes[0], _)

        axes[1].imshow(ct, cmap='gray')
        _ = axes[1].imshow(pd, cmap='jet', alpha=0.6)
        add_colorbar(figs, axes[1], _)

        #  plt.show()
        plt.savefig(f'test/{time}/{uid}_slice{i}_{md.max()}.png')
        print(f'save test images: test/{time}/{uid}_slice{i}_{md.max()}.png')
        plt.close()
    print('plot done')

def test_plot_bak(beam_id, CTs, mcDose, pbDose, dose_grid):
    '''dose_grid: (#grid_points, 3=(x,y,z)) @ 126x512x512'''
    dose_grid = (dose_grid/2.).round()  # @ 63x256x256
    dose_grid[:,:2] = dose_grid[:,:2] - 128/2 # account for center crop 256x256->128x128 

    for i, (ct, md, pd) in enumerate(zip(CTs, mcDose, pbDose)): 
        slice_index    = np.where(dose_grid[:,-1]==i)[0]
        slice_doseGrid = dose_grid[slice_index]

        figs, axes = plt.subplots(1, 3, figsize=(3*10, 1*10))
        figs.set_tight_layout(True)

        axes[0].imshow(ct, cmap='gray')
        _ = axes[0].imshow(md, cmap='jet', alpha=0.6)
        add_colorbar(figs, axes[0], _)

        axes[1].imshow(ct, cmap='gray')
        _ = axes[1].imshow(pd, cmap='jet', alpha=0.6)
        add_colorbar(figs, axes[1], _)

        axes[2].imshow(ct, cmap='gray')
        axes[2].scatter(slice_doseGrid[:,0], slice_doseGrid[:,1], s=0.3)

        plt.show()
        plt.savefig(f'test/test_{i}_{beam_id}_{md.max()}.png')
        print(f'save test images: test/test_{i}_{beam_id}_{md.max()}.png')
        plt.close()
    print('plot done')

def test_save_result(mp):
    def display():
        doses = cal_dose(mp.optimizer.deposition, fluence) # cal dose (#voxels, )
        dict_organ_doses = split_doses(doses, mp.data.organName_ptsNum)  # split organ_doses to obtain individual organ doses
        loss, breaking_points_nums = mp.optimizer.loss.loss_func(dict_organ_doses)
        print(f'breaking points #: ', end='')
        for organ_name, breaking_points_num in breaking_points_nums.items(): print(f'{organ_name}: {breaking_points_num}   ', end='')
        print(f'loss={to_np(loss)}\n\n')

    # ndarray to tensor
    dict_segments, dict_MUs, dict_partialExp = OrderedBunch(), OrderedBunch(), OrderedBunch()
    for (beam_id, MU), (_, seg), (_, pe) in zip(mp.dict_MUs.items(), mp.dict_segments.items(), mp.dict_partialExp.items()):
        dict_segments[beam_id]   = torch.tensor(seg, dtype=torch.float32, device='cpu')
        dict_MUs[beam_id]        = torch.tensor(MU,  dtype=torch.float32, device='cpu', requires_grad=True)
        dict_partialExp[beam_id] = torch.tensor(pe,  dtype=torch.float32, device='cpu', requires_grad=True)

    fluence = mp.optimizer.computer_fluence(dict_segments, dict_partialExp, mp.dict_lrs, dict_MUs)[0] # (#valid_bixels,)
    display()

    def _test(is_seg_modulate):
        def _modulate_segment_with_partialExposure(seg, lrs, pes):
            '''
            Imposing the partialExp effect at the endpoint of leaf
            lrs: (#aperture, H, 2); seg:(HxW, #aperture); pes:(#aperture, H, 2)
            '''
            for i, aperture in enumerate(lrs):  # for each aperture
                for j, lr in enumerate(aperture):  # for each row
                    [l, r] = lr
                    l_pe, r_pe = sigmoid(pes[i, j])
                    # close hopeless bixel?
                    if l_pe < 0.6:
                        #  seg[j*W:j*W+W, i] [l] = 0
                        seg[j*W+l, i] = 0
                    if r_pe < 0.6:
                        #  seg[j*W:j*W+W, i] [r-1] = 0
                        seg[j*W+(r-1), i] = 0
            return seg

        dict_segments, dict_MUs = OrderedBunch(), OrderedBunch()
        for (beam_id, MU), (_, seg) in zip(mp.dict_MUs.items(), mp.dict_segments.items()):
            H, W = mp.data.dict_rayBoolMat[beam_id].shape
            
            validRay = mp.data.dict_rayBoolMat[beam_id].flatten().reshape((-1,1)) # where 1 indicates non-valid/blocked bixel 
            validRay = np.tile(validRay, (1, seg.shape[1]))  # (HxW, #aperture)
            seg = seg*validRay  # partialExp may open bixels in non-valid regions.

            lrs = dict_lrs[beam_id]  # (#aperture, H, 2)
            pes = dict_partialExp[beam_id] # (#aperture, H, 2)
            if is_seg_modulate:
                seg = _modulate_segment_with_partialExposure(seg, lrs, pes)

            dict_segments[beam_id] = torch.tensor(seg, dtype=torch.float32, device='cpu')
            dict_MUs[beam_id]      = torch.tensor(MU,  dtype=torch.float32, device='cpu', requires_grad=True)

    _test(is_seg_modulate=False)
    fluence = computer_fluence(mp.data, dict_segments, dict_MUs)[0]
    display()

    _test(is_seg_modulate=True)
    fluence = computer_fluence(mp.data, dict_segments, dict_MUs)[0]
    display()

def restore_lrs(segs, H, W):
    aps = segs.shape[-1]
    lrs = np.zeros((aps, H, 2), np.int)
    for ap in range(aps):
        seg = segs[:,ap].reshape(H, W)
        for h in range(H):
            row = seg[h]
            if (row==0).all():  # closed row
                lrs[ap, h] = np.array([0, 0])
            else:
                first, last = np.nonzero(row)[0][[0,-1]]  # get first 1 and last 1 positions
                lrs[ap, h] = np.array([first, last+1])
    return lrs

if __name__ == "__main__":
    _test_smallest_contiguous_sum()
