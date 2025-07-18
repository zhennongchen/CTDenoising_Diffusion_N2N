import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.noise2noise.Generator as Generator

#######################
trial_name = 'noise2noise_2D'
epoch = 78
trained_model_filename = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'pred_images'); os.makedirs(save_folder, exist_ok=True)

image_size = [512,512]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation' 
#######################
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_NAS.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)


# build model
model = noise2noise.Unet2D(
    init_dim = 16,
    channels = 2, 
    out_dim = 1,
    dim_mults = (2,4,8,16),
    full_attn = (None,None, False, True),
    act = 'ReLU',
)

# main
for i in range(0, n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_num = random_num_list[n[i]]
    x0_file = x0_list[n[i]]
    condition_file = condition_list[n[i]]

    print(i,patient_id, patient_subid, random_num)

    # get the ground truth image
    gt_img = nb.load(x0_file)
    affine = gt_img.affine; gt_img = gt_img.get_fdata()[:,:,30:80]

    # get the condition image
    condition_img = nb.load(condition_file).get_fdata()[:,:,30:80]

    # make folders
    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num))])
    save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'epoch' + str(epoch)); os.makedirs(save_folder_case, exist_ok=True)

    # # generator
    generator = Generator.Dataset_2D(
        img_list = np.array([condition_file]),
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = False,
        slice_range = [30,80],

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)

    # # sample:
    sampler = noise2noise.Sampler(model,generator,batch_size = 1, image_size = image_size)

    pred_img = sampler.sample_2D(trained_model_filename, gt_img)
    pred_img_final = pred_img
    # pred_img_final = np.zeros(gt_img.shape)
    # pred_img_final[:,:,0] = gt_img[:,:,0]
    # pred_img_final[:,:,1:pred_img_final.shape[-1]-1] = pred_img[:,:,0:pred_img_final.shape[-1]-2]
    # pred_img_final[:,:,-1] = gt_img[:,:,-1]
    
    # save
    nb.save(nb.Nifti1Image(pred_img_final, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
#     nb.save(nb.Nifti1Image(gt_img, affine), os.path.join(save_folder_case, 'gt_img.nii.gz'))
#     nb.save(nb.Nifti1Image(condition_img, affine), os.path.join(save_folder_case, 'condition_img.nii.gz'))
# # 
    # save_folder_case = os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num), 'final_avg'); os.makedirs(save_folder_case, exist_ok=True)
    # folders = ff.find_all_target_files(['epoch*'], os.path.join(save_folder, patient_id, patient_subid, 'random_' + str(random_num)))
    # final_pred = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[-1], len(folders)))
    # for j in range(len(folders)):
    #     pred_img = nb.load(os.path.join(folders[j], 'pred_img.nii.gz')).get_fdata()
    #     final_pred[:,:,:,j] = pred_img
    # final_pred = np.mean(final_pred, axis = -1)
    # nb.save(nb.Nifti1Image(final_pred, affine), os.path.join(save_folder_case, 'pred_img.nii.gz'))
