'''
Baseline fanbeam reconstruction
'''

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import SimpleITK as sitk
import scipy.ndimage

import ct_projector.projector.numpy as ct_projector
import ct_projector.projector.numpy.parallel as ct_para

from locations import data_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser(description='FBP reconstruction')
    parser.add_argument('--input_dir')
    parser.add_argument('--geometry')
    parser.add_argument('--output_dir')

    parser.add_argument('--nrot_per_slab', type=int, default=1)
    parser.add_argument('--start_rot', type=int, default=0)
    parser.add_argument('--end_rot', type=int, default=-1)
    parser.add_argument('--nz_per_slice', type=int, default=1)

    parser.add_argument('--filter', default='hann')
    parser.add_argument('--filter_multiply_rl', type=int, default=1)
    parser.add_argument('--filter_len', type=int, default=2048)
    parser.add_argument('--filter_zoom', type=float, default=1)

    parser.add_argument('--device', type=int, default=0)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
        args.debug = True
    else:
        args = parser.parse_args()
        args.debug = False

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def read_projection_data(
    input_dir, projector: ct_projector.ct_projector, start_view, end_view, nrot_per_slab, nz_per_slice
):
    min_file_bytes = 1024 * 10  # file size should be at least 10MB

    if end_view < 0:
        end_view = len(glob.glob(os.path.join(input_dir, 'slab_*.nii.gz'))) - 1

    filenames = []
    for iview in range(start_view, end_view + 1):
        filename = os.path.join(input_dir, f'slab_{iview}.nii.gz')
        if not os.path.exists(filename):
            break
        if os.path.getsize(filename) < min_file_bytes:
            break
        filenames.append(filename)

    prjs = []
    print('Reading data from {0} files'.format(len(filenames)), flush=True)
    for i, filename in enumerate(filenames):
        print(i, end=',', flush=True)
        prj = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        prj = prj.astype(np.float32)
        prjs.append(prj)
    print('')
    prjs = np.array(prjs)

    prjs = prjs.reshape([
        nrot_per_slab,
        prjs.shape[0] // nrot_per_slab,
        prjs.shape[1],
        prjs.shape[2] // nz_per_slice,
        nz_per_slice,
        prjs.shape[3],
    ])

    prjs = np.mean(prjs, axis=(0, 4))

    # update projector
    projector.nv = prjs.shape[2]
    projector.nz = prjs.shape[2]
    projector.dv = projector.dv * nz_per_slice
    projector.dz = projector.dv

    return prjs, projector


# %%
def load_custom_filter(
    projector: ct_projector.ct_projector,
    prjs,
    filter,
    filter_len,
    filter_zoom,
    filter_multiply_rl,
):
    custom_filter = os.path.join(data_dir, filter)
    if os.path.exists(custom_filter):
        print('Loading custom filter from file', flush=True)
        custom_filter = np.fromfile(custom_filter, np.float32)
        if filter_len > 0 and len(custom_filter) != filter_len:
            new_filter = custom_filter[:len(custom_filter) // 2]
            new_filter = scipy.ndimage.zoom(new_filter, filter_zoom, order=1)
            custom_filter = np.zeros([filter_len], np.float32)
            custom_filter[:len(new_filter)] = new_filter
        custom_filter[len(custom_filter) // 2:] = custom_filter[len(custom_filter) // 2:0:-1]
        plt.plot(custom_filter)
    else:
        print('Using build-in filters', flush=True)
        custom_filter = None

    nu = prjs.shape[-1]
    du = projector.du
    rl_filter = np.zeros([2 * nu - 1], np.float32)
    k = np.arange(len(rl_filter)) - (nu - 1)
    for i in range(len(rl_filter)):
        if k[i] == 0:
            rl_filter[i] = 1 / (4 * du * du)
        elif k[i] % 2 != 0:
            rl_filter[i] = -1 / (np.pi * np.pi * k[i] * k[i] * du * du)
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter)

    frl_filter = frl_filter * len(frl_filter) / prjs.shape[1] * du * 2

    if filter_multiply_rl:
        custom_filter = frl_filter * custom_filter

    return custom_filter


# %%
def main(args):
    ct_projector.set_device(args.device)

    input_dir = os.path.join(data_dir, args.input_dir)
    geometry = os.path.join(data_dir, args.geometry)
    output_dir = os.path.join(data_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('Reading geometry...', flush=True)
    projector = ct_projector.ct_projector()
    projector.from_file(geometry)
    angles = projector.get_angles() / 2

    print('Reading data...', flush=True)
    prjs, projector = read_projection_data(
        input_dir, projector, args.start_rot, args.end_rot, args.nrot_per_slab, args.nz_per_slice
    )

    for k in vars(projector):
        print(k, '=', getattr(projector, k), flush=True)

    custom_filter = load_custom_filter(
        projector, prjs, args.filter, args.filter_len, args.filter_zoom, args.filter_multiply_rl
    )

    print('Reconstructing...', flush=True)
    prjs = np.copy(prjs, 'C')
    if custom_filter is None:
        fprjs = ct_para.ramp_filter(projector, prjs, args.filter)
    else:
        # 让filter的两边保持对称
        fprjs = np.fft.fft(prjs, len(custom_filter), axis=-1)
        fprjs = fprjs * custom_filter
        fprjs = np.fft.ifft(fprjs, axis=-1)[..., :prjs.shape[-1]]
        fprjs = fprjs.real.astype(np.float32) * np.pi / len(custom_filter) / 2
        fprjs = np.copy(fprjs, 'C')
    recon = ct_para.pixel_driven_bp(projector, fprjs, angles)

    recon = recon / 0.0193 * 1000 - 1000

    recon = recon.reshape([-1, recon.shape[2], recon.shape[3]])

    if args.debug:
        plt.figure(figsize=[16, 8])
        plt.subplot(121)
        plt.imshow(recon[recon.shape[0] // 2, 96:-96, 96:-96], 'gray', vmin=-400, vmax=2000)
        plt.subplot(122)
        plt.imshow(recon[recon.shape[0] // 2, 96:-96, 96:-96], 'gray', vmin=0, vmax=100)

    print('Save results...', flush=True)
    recon = recon.astype(np.int16)
    sitk_recon = sitk.GetImageFromArray(recon)
    sitk_recon.SetSpacing([float(projector.dx), float(projector.dy), float(projector.dz)])
    sitk.WriteImage(sitk_recon, os.path.join(output_dir, 'recon.nii.gz'))

    print('All done')

    return prjs, recon


# %%
if __name__ == '__main__':
    args = get_args([
        '--input_dir', '20230518_phantom_and_pig_raw/reconstruction/head_phantom/prjs',
        '--geometry', 'pcd_rebinned.cfg',
        '--output_dir', '20230518_phantom_and_pig_raw/reconstruction/head_phantom/bone_0.707mm',

        '--start_rot', '0',
        '--end_rot', '-1',

        '--nz_per_slice', '1',
        # '--filter', 'rl',
        '--filter', '20220720_1x1/processed/additional_bone_kernel',
        '--filter_len', '2048',
    ])

    res = main(args)
