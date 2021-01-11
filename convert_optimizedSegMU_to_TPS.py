from utils import *
from data import Data
from options import BaseOptions
from termcolor import colored, cprint


def main(hparam):

    data = Data(hparam)
    fn = os.path.join(hparam.optimized_segments_MUs_file_path, 'optimized_segments_MUs.pickle')
    dict_seg_mu = unpickle_object(fn)
    cprint(f'using optimized_segments_MUs from file: {fn}', 'green')

    for beamid, dict_beam_seg_mu in dict_seg_mu.items():
        file_write_obj = open(os.path.join(hparam.optimized_segments_MUs_file_path, f'{beamid}.txt'), 'w')
        seg, mu = dict_beam_seg_mu['Seg'], dict_beam_seg_mu['MU']
        high, width = data.dict_rayBoolMat[beamid].shape
        cprint(f'beam_id={beamid}', 'green')
        cprint(f'1d seg shape: {seg.shape}', 'green')
        cprint(f'2d seg shape: {high} x {width}', 'green')
        cprint(f'mu shape: {mu.shape}', 'green')

        for col_idx in range(seg.shape[1]):
            file_write_obj.writelines(str(col_idx)+':'+'\r\n')
            file_write_obj.writelines('mu:'+str(mu[col_idx])+'\r\n')
            seg_2d = seg[:, col_idx].astype(np.uint8).reshape(high, width)
            #  ibegin,iend = 0,0  # Juyao, Bug
            for iRow in range(high):
                ibegin,iend = 0,0  #  should be reset to 0 for each row
                #print('\n')
                for iCol in range(width):
                    if seg_2d[iRow,iCol]>0:
                        ibegin=iCol
                        #print(f'iRow={iRow};iCol={iCol};value={seg_2d[iRow,iCol]};ibegin={ibegin}')
                        break;
                for iCol in range(width-1,-1,-1):
                    if seg_2d[iRow, iCol] > 0:
                        iend = iCol
                        #print(f'iRow={iRow};iCol={iCol};value={seg_2d[iRow,iCol]};iend={iend}')
                        break;
                file_write_obj.writelines('row:' + str(iRow) +'\t'+ str(ibegin)+'\t' +str(iend) +'\r\n')
            # pdb.set_trace()
            # print(seg_2d.shape)
        file_write_obj.close()

if __name__ == "__main__":
    hparam = BaseOptions().parse()
    main(hparam)

