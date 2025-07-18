import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys
import numpy as np 
import pandas as pd
import nibabel as nb
from skimage.measure import block_reduce

import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

data_path = '/mnt/camca_NAS/Portable_CT_data'
save_path = '/mnt/camca_NAS/denoising/Data'

patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/denoising/','Patient_lists', 'fixedCT_static.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
print('patient sheet len: ', len(patient_sheet))

for i in range(0, len(patient_sheet)):
    row = patient_sheet.iloc[i]
    patient_id = row['Patient_ID']
    patient_subID = row['Patient_subID']
    use = row['use']

    original_file = os.path.join(data_path,'nii_imgs_202404/static',patient_id,patient_subID,'fixed', use+'.nii.gz')

    if os.path.isfile(os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_thinslice.nii.gz')):
        print('already processed: ', patient_id, patient_subID)
        
    else:
        print('processing: ', patient_id, patient_subID)
    
        # get the affine and pixel dimension
        img = nb.load(original_file)
        affine = img.affine
        pixdim = img.header.get_zooms()
        img_data = img.get_fdata()
        
        # ### [1,1, original_z]
        # # turn x and y dim into 1mm
        # scale_factor = [int(1/pixdim[0]), int(1/pixdim[1]), 1]
        # # use block_reduce to downsample the image
        # img_data = img.get_fdata()
        # img_data_xy1mm = block_reduce(img_data, tuple(scale_factor), np.mean)

        # # change the affine and pixel dimension 
        # new_affine = affine.copy()
        # new_affine[0, 0] *= scale_factor[0]
        # new_affine[1, 1] *= scale_factor[1]

        # # new pixeldim
        # new_pixdim = (pixdim[0]*scale_factor[0], pixdim[1]*scale_factor[1], pixdim[2])
        # # save in the header
        # img.header.set_zooms(new_pixdim)

        # # save the image
        ff.make_folder([os.path.join(save_path, 'fixedCT', patient_id),os.path.join(save_path, 'fixedCT', patient_id, patient_subID)])
        # save_file = os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_xy1mm.nii.gz')
        # nb.save(nb.Nifti1Image(img_data_xy1mm, new_affine, img.header), save_file)


        ### 5mm
        z_scale_factor = int(5 // pixdim[2])
        print('z_scale_factor: ', z_scale_factor)
        img_data_xyz5mm = block_reduce(img_data, (1,1,z_scale_factor), np.mean)

        # change affine and pixel dimension accordingly
        new_affine_5mm = affine.copy()
        new_affine_5mm[2, 2] *= z_scale_factor
        new_pixdim_5mm = (pixdim[0],pixdim[1], pixdim[2]*z_scale_factor)
        # save in the header
        img.header.set_zooms(new_pixdim_5mm)

        # save the image
        save_file = os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_5mm.nii.gz')
        nb.save(nb.Nifti1Image(img_data_xyz5mm, new_affine_5mm, img.header), save_file)

        ### interpolate to 1.25mm
        new_dim = [pixdim[0], pixdim[1], 0.625]

        img_5mm = nb.load(os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_5mm.nii.gz'))
        hr_resample = Data_processing.resample_nifti(img_5mm, order=1,  mode = 'nearest',  cval = np.min(img_5mm.get_fdata()), in_plane_resolution_mm=new_dim[0], slice_thickness_mm=new_dim[-1])
        nb.save(hr_resample, os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_thinslice.nii.gz'))

    if os.path.isfile(os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_thinslice_partial.nii.gz')) == 1:
        print('done partial')
    else:
        # pick only middle 100 slices in thinslice
        img_thinslice = nb.load(os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_thinslice.nii.gz'))
        img_thinslice_data = img_thinslice.get_fdata()[:,:,img_thinslice.shape[2]//2-50:img_thinslice.shape[2]//2+50]
        nb.save(nb.Nifti1Image(img_thinslice_data, img_thinslice.affine, img_thinslice.header), os.path.join(save_path, 'fixedCT', patient_id, patient_subID, 'img_thinslice_partial.nii.gz'))
   


