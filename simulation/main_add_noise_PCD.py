#!/usr/bin/env python

# %%
import numpy as np
import nibabel as nb 
import pandas as pd
import os
import copy
import Diffusion_denoising_thin_slice.simulation.ct_basic_PCD as ct
import Diffusion_denoising_thin_slice.functions_collection as ff
import ct_projector.projector.numpy.parallel as ct_para

main_path = '/mnt/camca_NAS/denoising/Data'


# set a noise range
noise_sheet = pd.read_excel(os.path.join(os.path.dirname(main_path),'Patient_lists','portable_CT_CNR.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
CNR_array = noise_sheet['CNR'].values
std_array = noise_sheet['background_std'].values
print('CNR max, min, mean, std, median, lower quartile, upper quartile: ', np.max(CNR_array), np.min(CNR_array), np.mean(CNR_array), np.std(CNR_array), np.median(CNR_array), np.percentile(CNR_array, 25), np.percentile(CNR_array, 75))
print('std max, min, mean, std, median, lower quartile, upper quartile: ', np.max(std_array), np.min(std_array), np.mean(std_array), np.std(std_array), np.median(std_array), np.percentile(std_array, 25), np.percentile(std_array, 75))

# CNR range: 5-8.6, mean+std = 6.6+-0.7, [6.1,6.9]
# background noise range: 4-22.8, mean+std = 10.5+-3, [8.8, 11.9]

# define the patient list
patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/','Patient_lists', 'fixedCT_static.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
print('patient sheet len: ', len(patient_sheet))

for i in range(0,len(patient_sheet)//3):
    row = patient_sheet.iloc[i]
    patient_id = row['Patient_ID']
    patient_subID = row['Patient_subID']

    print(patient_id, patient_subID)

    save_folder_case = os.path.join(main_path,'simulation', patient_id,patient_subID)
    ff.make_folder([os.path.dirname(save_folder_case), save_folder_case])

    # for dose factor in range 0.30-0.9 with step 0.05
    # possion_hann_dose_range = [0.60,0.80]
    # gaussian_custom_dose_range = [0.085,0.1]

    # img file
    img_file = os.path.join(main_path,'fixedCT',patient_id,patient_subID,'img_thinslice_partial.nii.gz')
    print(img_file)
    # load img
    img_clean = nb.load(img_file).get_fdata().astype(np.float32)
    img_clean[img_clean < -1024] = -1024
    spacing = nb.load(img_file).header.get_zooms()[::-1]
    affine = nb.load(img_file).affine

    for noise_type in ['possion', 'gaussian']:
        # by vusialization we decided the following dose range
        possion_hann_dose_range = [0.10,0.20]
        gaussian_custom_dose_range = [0.15,0.25]
        
        for k in range(0,2):
            save_folder_k = os.path.join(save_folder_case, noise_type+'_random_'+str(k));ff.make_folder([save_folder_k])
            if os.path.isfile(os.path.join(save_folder_k,'recon.nii.gz')):
                print('already done, continue')
                continue

            if noise_type == 'possion':
                dose_factor = np.random.uniform(possion_hann_dose_range[0],possion_hann_dose_range[1] + 1e-8)
            elif noise_type == 'gaussian':
                dose_factor = np.random.uniform(gaussian_custom_dose_range[0],gaussian_custom_dose_range[1] + 1e-8)
            print('dose factor: ', dose_factor)

            # process img
            img0 = img_clean.copy()
            img0 = np.rollaxis(img0,-1,0)
            print('img shape, min, max: ', img0.shape, np.min(img0), np.max(img0))
            print('spacing: ', spacing)
      
            # define projectors
            projector = ct.define_forward_projector_pcd(img0,spacing, file_name = './pcd_parallel_6x5_512.cfg')

            # FP
            # set angles
            angles = projector.get_angles()

            recon_noise = np.zeros((img0.shape[1], img0.shape[2], img0.shape[0]), np.float32)
            for slice_n in range(0, img0.shape[0]):
                img_slice = img0[[slice_n],:,:].copy()
                img_slice = (img_slice[np.newaxis, ...] + 1000) / 1000 * 0.019 

                prjs = ct_para.distance_driven_fp(projector, img_slice, angles)
                fprjs = ct_para.ramp_filter(projector, prjs, 'rl')

                # add noise
                if noise_type[0:2] == 'po':
                    # add poisson noise
                    noise_of_prjs = ct.add_poisson_noise(prjs, N0=1000000, dose_factor = dose_factor) - prjs
                elif noise_type[0:2] == 'ga':
                    # add gaussian noise
                    noise_of_prjs = ct.add_gaussian_noise(prjs, N0=1000000, dose_factor = dose_factor) - prjs

                # recon
                if noise_type[0:2] == 'po':
                    # hann filter
                    fnoise = ct_para.ramp_filter(projector, noise_of_prjs, 'hann')
                    recon_hann = ct_para.distance_driven_bp(projector, fnoise, angles, True) + img_slice
                    recon_hann = recon_hann[0, 0] / 0.019 * 1000 - 1000
                    recon_noise[:,:,slice_n] = recon_hann
                
                elif noise_type[0:2] == 'ga':
                    # custom filter
                    custom_additional_filter = ct.get_additional_filter_to_rl(os.path.join('/mnt/camca_NAS/denoising/Data', 'softTissueKernel_65'), projector.nu, projector.du, projector.nview)
                    recon_custom = ct.interleave_filter_and_recon(projector, noise_of_prjs, custom_additional_filter,angles) + img_slice
                    recon_custom = recon_custom[0, 0] / 0.019 * 1000 - 1000
                    recon_noise[:,:,slice_n] = recon_custom
            # save recon
            # nb.save(nb.Nifti1Image(img_clean[:,:,30:60], affine), os.path.join(save_folder_case,'img_clean.nii.gz'))
            nb.save(nb.Nifti1Image(recon_noise, affine), os.path.join(save_folder_k,'recon.nii.gz'))

            