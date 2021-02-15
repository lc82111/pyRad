#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, pdb, glob, json
from pathlib import Path

from utils import *
from data import Data, Geometry
from options import BaseOptions


def get_argument_from_patient_0_bat(MonteCarlo_dir):
    args_dict = {} 
    with open(Path(MonteCarlo_dir, 'patient_0.bat'), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "gDPM" in line:
                args =['--GPU_ID', '--OutputPrefix', '--NumberOfHistories', '--PslFilePrefix', '--CalibrationFactor',
                       '--ElectronAbsorptionEnergy', '--PhotonAbsorptionEnergy', '--CompatibleFilePrefix', 
                       '--DoseToWaterConversionFile', '--EGS4GeometryFile',
                       '--CTdimension x', '--CTdimension y', '--CTdimension z',
                       '--CTresolution x', '--CTresolution y', '--CTresolution z',
                       '--CToffset x', '--CToffset y', '--CToffset z', 
                       '--Isocenter x', '--Isocenter y', '--Isocenter z',
                       '--SAD', '--BeamAnglesFile', '--JawsFile', '--FluenceMapFile', '--BBASmoothFunction', '--NumberOfEachBatch']
                argstrings = line.replace('"gDPM_v3.exe" ','').split(' > ')[0].split(' ')
                assert len(args) == len(argstrings)
                for (a, s) in zip(args, argstrings):
                    assert a.split(' ')[0] == s.split('=')[0]
                    args_dict[a.split('--')[1]] = s.split('=')[1]
                return args_dict

def main(hparam):
    data = Data(hparam)
    geometry = Geometry(data)
    args_dict = get_argument_from_patient_0_bat(hparam.MonteCarlo_dir)
    winServer_MonteCarloDir = Path(hparam.winServer_MonteCarloDir)
    
    # mkdir
    dirs = ['FM_results', 'gDPM_results', 'Segs', 'templates']
    for d in dirs:
        make_dir(winServer_MonteCarloDir.joinpath(d))

    # egs4phant 
    cp(Path(hparam.MonteCarlo_dir, 'patient.egs4phant'), winServer_MonteCarloDir.joinpath('patient.egs4phant'))
    
    ## templates
    # 0. gDPM_config.json 
    mcD, mcH, mcW = hparam.MCDose_shape
    assert mcD == int(args_dict['CTdimension z'])
    assert mcH == int(args_dict['CTdimension y'])
    assert mcW == int(args_dict['CTdimension x'])
    
    ctW, ctH, ctD = geometry.CT.size
    sx, sy, sz = geometry.CT.spacing/10  # mm->cm 
    sz, sy, sx = sz*ctD/mcD, sy*ctH/mcH, sx*ctW/mcW 
    sz, sy, sx = round(sz,7), round(sy,7), round(sx,7)   
    mcsz, mcsy, mcsx = float(args_dict['CTresolution z']), float(args_dict['CTresolution y']), float(args_dict['CTresolution x'])
    if mcsx!=sx: cprint(f'[Warning] patient.bat --CTresolution x {mcsx} != CT DICOM spacing x {sx}', 'red')  
    if mcsy!=sy: cprint(f'[Warning] patient.bat --CTresolution y {mcsy} != CT DICOM spacing y {sy}', 'red')  
    if mcsz!=sz: cprint(f'[Warning] patient.bat --CTresolution z {mcsz} != CT DICOM spacing z {sz}', 'red')  

    ox, oy, oz = geometry.CT.origin/10  # mm->cm
    #  oz, oy, ox = round(oz,7), round(oy,7), round(ox,7)
    ctoffset_x = float(args_dict['CToffset x']) 
    ctoffset_y = float(args_dict['CToffset y']) 
    ctoffset_z = float(args_dict['CToffset z']) 
    if ox != ctoffset_x: cprint(f'[Warning] patient.bat --CToffset x {ctoffset_x} != CT DICOM origin x {ox}', 'red')  
    if oy != ctoffset_y: cprint(f'[Warning] patient.bat --CToffset y {ctoffset_y} != CT DICOM origin y {oy}', 'red')  
    if oz != ctoffset_z: cprint(f'[Warning] patient.bat --CToffset z {ctoffset_z} != CT DICOM origin z {oz}', 'red')  # NOTE: inconsistent

    cx, cy, cz = geometry.plan.isocenter/10  # mm->cm
    iso_x = float(args_dict['Isocenter x'])
    iso_y = float(args_dict['Isocenter y'])
    iso_z = float(args_dict['Isocenter z'])
    if cx != iso_x: cprint(f'[Warning] patient.bat --Isocenter x {iso_x} != RTPlan DICOM isocenter x {cx}', 'red')  
    if cy != iso_y: cprint(f'[Warning] patient.bat --Isocenter y {iso_y} != RTPlan DICOM isocenter y {cy}', 'red')  
    if cz != iso_z: cprint(f'[Warning] patient.bat --Isocenter z {iso_z} != RTPlan DICOM isocenter z {cz}', 'red')   # NOTE: inconsistent

    gDPM_config =  { "CTdimension": {
                        "x": mcW,
                        "y": mcH,
                        "z": mcD, 
                      },
                      "CTresolution": {
                        "x": sx, #0.234375 cm
                        "y": sy, #0.234375 cm
                        "z": sz, #0.25     cm
                      },
                      "CToffset": {
                        "x": ctoffset_x, #-30    cm
                        "y": ctoffset_y, #-30    cm
                        "z": ctoffset_z, #-15.75 cm
                      },
                      "Isocenter": {
                        "x": iso_x, # -1.809999
                        "y": iso_y, #-0.1599998
                        "z": iso_z, #5.379999
                      },
                      "SAD": float(args_dict['SAD']),
                      "NumberOfHistories": 10000000112,
                      "NumberOfEachBatch": 10000000112
                    }

    json.dump(gDPM_config, open(winServer_MonteCarloDir.joinpath('templates', 'gDPM_config.json'), "w"), indent=4) 

    # 1. FM_info.txt
    fm_info_path =glob.glob(str(Path(hparam.MonteCarlo_dir, 'FM_info*.txt')))[0]
    with open(fm_info_path, "r") as f:
        lines = f.readlines()
        flag = False
        for i, line in enumerate(lines):
            if 'numControlPoints' in line:
                flag = True
                continue
            if flag: 
                lines[i] = '2\n'  # set aperture number = 1
                break
    with open(winServer_MonteCarloDir.joinpath('templates', 'FM_info.txt'), "w") as f:
        f.writelines(lines)

    # 2. Seg_beamIDx.txt
    seg_fns = glob.glob(str(Path(hparam.MonteCarlo_dir, 'Seg*.txt')))
    seg_fns.sort() # sort in place
    assert len(seg_fns) == geometry.plan.beam_numbers
    for i, fn in enumerate(seg_fns):
        with open(fn, "r") as f:
            lines = f.readlines()[:7]
            # set MU to 1.0
            line5 = lines[5].split()
            line5[0] = '1.0'
            new_line5 = ' '.join(line5) + '\n'
            lines[5] = new_line5
            lines.append('2\n')
            lines.append('0\n')
            lines.append('1.0\n')
        new_fn = winServer_MonteCarloDir.joinpath('templates', f'Seg_beamID{i+1}.txt')
        with open(new_fn, "w") as f:
            f.writelines(lines)
    cprint(f'Done. All files have been written to {winServer_MonteCarloDir}.', 'green')

if __name__ == "__main__":
    hparam = BaseOptions().parse()
    main(hparam)
