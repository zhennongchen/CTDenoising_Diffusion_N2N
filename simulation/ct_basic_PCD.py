#!/usr/bin/env python

import ct_projector.projector.numpy.parallel as ct_para

import numpy as np
import cupy as cp
import copy
import nibabel as nb

# function: generate angle list
def get_angles_zc(nview, total_angle,start_angle):
    return np.arange(0, nview, dtype=np.float32) * (total_angle / 180 * np.pi) / nview + (start_angle / 180 * np.pi)


# for nibabel:
def basic_image_processing(filename , convert_value = True, header = False):
    ct = nb.load(filename)
    spacing = ct.header.get_zooms()
    img = ct.get_fdata()
    
    if convert_value == True:
        img = (img.astype(np.float32) + 1024) / 1000 * 0.019 
        img[img < 0] = 0
        
    img = np.rollaxis(img,-1,0)

    spacing = np.array(spacing[::-1])

    if header == False:
        return img,spacing,ct.affine
    else:
        return img, spacing, ct.affine, ct.header


def define_forward_projector_pcd(img,spacing, file_name = './pcd_parallel_6x5_512.cfg'):
    projector = ct_para.ct_projector()
    projector.from_file(file_name)
    projector.dx = spacing[1]
    projector.dy = spacing[2]
    projector.dz = spacing[0]
    projector.nx = img.shape[1]
    projector.ny = img.shape[2]
    projector.dv = projector.dz
    # print('[projector]')
    # for k in vars(projector):
    #     print(k, '=', getattr(projector, k), flush=True)
    return projector


def add_poisson_noise(prj: np.ndarray, N0: float, dose_factor: float, seed: int=None):

    if seed is not None:
        np.random.seed(seed)
    else:
        rand_int = np.random.randint(0, 1000000001) 
        np.random.seed(rand_int)

    if N0 > 0 and dose_factor < 1:
        # Convert log projection back to photon counts
        I_noisy = np.random.poisson(lam=(N0 * dose_factor * np.exp(-prj)))

        # Prevent log(0) issues by setting minimum value
        I_noisy = np.maximum(I_noisy, 1)

        # Convert back to log domain
        prj_noisy = -np.log(I_noisy / (N0 * dose_factor))

        return prj_noisy.astype(np.float32)

    return prj.astype(np.float32)

def add_gaussian_noise(prj: np.ndarray, N0: float, dose_factor: float,seed: int=None):
    if seed is not None:
        np.random.seed(seed)
    else:
        rand_int = np.random.randint(0, 1000000001) 
        np.random.seed(rand_int)

    # add noise
    if N0 > 0 and dose_factor < 1:
        prj = prj + np.sqrt((1 - dose_factor) / dose_factor * np.exp(prj) / N0) * np.random.normal(size=prj.shape)
        prj = prj.astype(np.float32)

    return prj

def get_additional_filter_to_rl(filename, nu, du, nview, ninterp=20):
    '''
    Compose the filter that is on top of the RL filter
    '''
    custom_filter = np.fromfile(filename, np.float32)

    # these are the parameters after interleaving
    # the filter will be applied on the interleaved data
    nu *= 2
    du /= 2

    # compose rl filter
    rl_filter = np.zeros([2 * nu - 1], np.float32)
    k = np.arange(len(rl_filter)) - (nu - 1)
    for i in range(len(rl_filter)):
        if k[i] == 0:
            rl_filter[i] = 1 / (4 * du * du)
        elif k[i] % 2 != 0:
            rl_filter[i] = -1 / (np.pi * np.pi * k[i] * k[i] * du * du)
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter)

    frl_filter = frl_filter * len(frl_filter) / nview * du * 2

    ratio = custom_filter / frl_filter

    # A smooth transition from 1 to the target ratio to avoid instability at the center
    ratio_fix = ratio.copy()
    ratio_fix[:ninterp] = 1 + (ratio[ninterp] - 1) * np.arange(ninterp) / ninterp
    ratio_fix = ratio_fix.astype(np.float32)
    ratio_fix[len(ratio_fix) // 2:] = ratio_fix[len(ratio_fix) // 2:0:-1]

    # plt.figure()
    # plt.plot(frl_filter)
    # plt.plot(custom_filter)
    # plt.plot(ratio)
    # plt.plot(ratio_fix)
    # plt.show()

    return ratio_fix

def interleave_filter_and_recon(projector, prjs, custom_filter,angles):
    # interleave the filter
    projector = copy.deepcopy(projector)
    prjs = prjs.copy()

    # make sure the detector center is within the central pixel (usually 0.25 of a pixel)
    # so that when interleaving the detectors the length will be twice the original
    offu = int(projector.off_u)
    projector.off_u -= offu
    projector.nu -= 2 * offu
    prjs = prjs[..., :-2 * offu]

    # interleave the projections
    new_prjs = np.zeros(list(prjs.shape[:-1]) + [prjs.shape[-1] * 2])
    for iview in range(new_prjs.shape[1]):
        iview_opp = (iview + new_prjs.shape[1] // 2) % new_prjs.shape[1]
        new_prjs[:, iview, :, 1::2] = prjs[:, iview, :, :]
        new_prjs[:, iview, :, 0::2] = prjs[:, iview_opp, :, ::-1]
    prjs = new_prjs
    projector.off_u = 0
    projector.nu = prjs.shape[-1]
    projector.du = projector.du / 2

    # build rl filter
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

    custom_filter = frl_filter * custom_filter

    # filter the projection
    fprj = np.fft.fft(prjs, len(custom_filter), axis=-1)
    fprj = fprj * custom_filter
    fprj = np.fft.ifft(fprj, axis=-1)[..., :prjs.shape[-1]]
    fprj = fprj.real.astype(np.float32) * np.pi / len(custom_filter) / 2
    fprj = np.copy(fprj, 'C')

    # reconstruction
    recon = ct_para.pixel_driven_bp(projector, fprj, angles)

    return recon

