'''
Insert noise into the brain CT simulation data
'''

# %%
import os
import sys
import argparse
import subprocess
import copy

import SimpleITK as sitk
import numpy as np
import pandas as pd

import ct_projector.projector.numpy as ct_projector
import ct_projector.projector.numpy.parallel as ct_para

from typing import Tuple

from brain_ct_ddpm.locations import data_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir')
    parser.add_argument('--input_manifest')
    parser.add_argument('--input_filename', default='img_5mm.nii.gz')
    parser.add_argument('--output_dir')
    parser.add_argument('--output_filename')

    parser.add_argument(
        '--geometry_filename',
        default='/workspace/brain_ct_ddpm/brain_ct_ddpm/prepare_brain_simulation/geometry.cfg'
    )

    parser.add_argument('--N0', type=float, default=1e6)
    parser.add_argument('--dose_factor', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--water_att', type=float, default=0.019)

    parser.add_argument('--device', type=int, default=0)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
    else:
        args = parser.parse_args()

    # get git hash
    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

    for k in vars(args):
        print(f'{k} = {getattr(args, k)}', flush=True)

    return args


# %%
def set_img_spacing(projector: ct_projector.ct_projector, spacing: Tuple[float, float, float]):
    projector = copy.deepcopy(projector)

    projector.dx = spacing[0]
    projector.dy = spacing[1]
    projector.dz = spacing[2]
    projector.dv = spacing[2]

    return projector


def forward_project(projector: ct_projector.ct_projector, img: np.array):
    img = img[:, np.newaxis, :, :]  # move z to the batch dimension
    angles = projector.get_angles()

    # forward projection
    proj = ct_para.distance_driven_fp(projector, img, angles)

    return proj


def reconstruction_interleave(projector: ct_projector.ct_projector, proj: np.ndarray, filter_type='rl'):
    '''
    Reconstruction with interleaved filtering
    '''
    projector = copy.deepcopy(projector)

    off_u = int(np.abs(projector.off_u))
    proj = proj[..., :-2 * off_u]

    interleaved_proj = np.zeros(list(proj.shape[:-1]) + [proj.shape[-1] * 2], np.float32)
    for iview in range(interleaved_proj.shape[1]):
        iview_opp = (iview + interleaved_proj.shape[1] // 2) % interleaved_proj.shape[1]
        interleaved_proj[:, iview, :, 1::2] = proj[:, iview, :, :]
        interleaved_proj[:, iview, :, 0::2] = proj[:, iview_opp, :, ::-1]

    projector.off_u = 0
    projector.nu = proj.shape[-1]
    projector.du = projector.du / 2
    angles = projector.get_angles()

    # filter
    fprj = ct_para.ramp_filter(projector, interleaved_proj, filter_type)
    recon = ct_para.distance_driven_bp(projector, fprj, angles, True)[:, 0, :, :]  # put z back to the z dimension

    return recon


def add_noise(prj: np.ndarray, N0: float, dose_factor: float, seed: int):
    np.random.seed(seed)

    # add noise
    if N0 > 0 and dose_factor < 1:
        prj = prj + np.sqrt((1 - dose_factor) / dose_factor * np.exp(prj) / N0) * np.random.normal(size=prj.shape)
        prj = prj.astype(np.float32)

    return prj


# %%
def main(args):
    ct_projector.set_device(args.device)

    input_dir = os.path.join(data_dir, args.input_dir)
    input_manifest = os.path.join(data_dir, args.input_manifest)
    geometry_filename = os.path.join(data_dir, args.geometry_filename)
    output_dir = os.path.join(data_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # avoid overwriting the input file
    assert (args.output_filename != args.input_filename) 

    manifest = pd.read_csv(input_manifest, dtype=str)

    # load geometry
    print('Reading geometry...', flush=True)
    projector = ct_projector.ct_projector()
    projector.from_file(geometry_filename)

    print('Adding noise {0}...'.format(len(manifest)), flush=True)
    for i, row in manifest.iterrows():
        print('Processing {0}: {1}...'.format(i + 1, row['ImageID']), flush=True)

        input_filename = os.path.join(input_dir, row['ImageID'], args.input_filename)

        # load the image
        sitk_img = sitk.ReadImage(input_filename)

        # convert from HU to attenuation
        img = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        img[img < -1024] = -1024
        img = (img + 1000) / 1000 * args.water_att

        # forward project
        fp_projector = set_img_spacing(projector, sitk_img.GetSpacing())
        proj = forward_project(fp_projector, img)

        # reconstruction
        recon = reconstruction_interleave(projector, proj, 'rl')

        # reconstruct noise only
        if args.dose_factor < 1:
            print('Adding noise...', flush=True)
            proj_noise = add_noise(proj, args.N0, args.dose_factor, args.seed) - proj
            recon_noise = reconstruction_interleave(projector, proj_noise, 'hann') # orignal data already has hann filter

            recon = recon + recon_noise

        # convert from attenuation to HU
        recon = (recon / args.water_att * 1000) - 1000
        sitk_recon = sitk.GetImageFromArray(recon.astype(np.int16))
        sitk_recon.CopyInformation(sitk_img)
        sitk_recon.SetSpacing([float(projector.dx), float(projector.dy), float(projector.dz)])

        # save the reconstruction
        output_filename = os.path.join(output_dir, row['ImageID'], args.output_filename)
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        sitk.WriteImage(sitk_recon, output_filename)

    print('All done')

    return sitk.GetArrayFromImage(sitk_img), proj, sitk.GetArrayFromImage(sitk_recon)


# # %%
# if __name__ == '__main__':
#     input_dir = './2022_mgh_brain_ct_dataset'
#     args = get_args([
#         '--input_dir', input_dir,
#         '--input_manifest', os.path.join(input_dir, 'manifest_data.csv'),
#         '--output_dir', input_dir,
#         '--dose_factor', '0.5',
#         '--output_filename', 'img_5mm_0.5.nii.gz'
#     ])

#     res = main(args)
