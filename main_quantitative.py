import sys
sys.path.append('/workspace/Documents')
import os
import torch
import lpips
import numpy as np 
import nibabel as nb
import pandas as pd
from scipy.ndimage import binary_erosion, generate_binary_structure
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.Data_processing as Data_processing

build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_NAS.xlsx'))
_,patient_id_list,patient_subid_list,random_num_list, condition_list, x0_list = build_sheet.__build__(batch_list = [5]) 
n = ff.get_X_numbers_in_interval(total_number = patient_id_list.shape[0],start_number = 0,end_number = 1, interval = 2)

def compute_lpips_3d(prediction, ground_truth,mask = None, max_val = None, min_val = None, net_type='vgg'):
    assert prediction.shape == ground_truth.shape, "Shape mismatch between prediction and ground truth!"
    
    # Convert to float32
    prediction = prediction.astype(np.float32)
    ground_truth = ground_truth.astype(np.float32)

    if mask is not None:
        prediction[mask==0] = np.min(prediction)
        ground_truth[mask==0] = np.min(ground_truth)

    # Normalize to [-1, 1] range as required by LPIPS
    if max_val == None:
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 2 - 1 
    else:
        prediction = (prediction - min_val) / (max_val - min_val) * 2 - 1
    if max_val == None:
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min()) * 2 - 1
    else:
        ground_truth = (ground_truth - min_val) / (max_val - min_val) * 2 - 1

    # Initialize LPIPS loss model
    loss_fn = lpips.LPIPS(net=net_type).to('cuda' if torch.cuda.is_available() else 'cpu')

    lpips_scores = []
    
    # Loop through each slice along the z-axis
    for i in range(prediction.shape[2]):
        pred_slice = prediction[:, :, i]  # Get 2D slice
        gt_slice = ground_truth[:, :, i]  # Get corresponding GT slice

        # Convert numpy arrays to torch tensors
        pred_tensor = torch.tensor(pred_slice).unsqueeze(0).unsqueeze(0)  # Shape: [1,1,H,W]
        gt_tensor = torch.tensor(gt_slice).unsqueeze(0).unsqueeze(0)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_tensor = pred_tensor.to(device)
        gt_tensor = gt_tensor.to(device)
        loss_fn = loss_fn.to(device)

        # Compute LPIPS score for this slice
        lpips_score = loss_fn(pred_tensor, gt_tensor)
        lpips_scores.append(lpips_score.item())

    # Compute average LPIPS score across all slices
    return np.mean(lpips_scores)
        
avg_slice = False # default
print('avg_slice:', avg_slice)
results = []; results_mean = []
for i in range(0,n.shape[0]):
    patient_id = patient_id_list[n[i]]
    patient_subid = patient_subid_list[n[i]]
    random_n = random_num_list[n[i]]
    print(patient_id, patient_subid, random_n)

    gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56_1/gt_img.nii.gz')
    gt_img = nb.load(gt_file).get_fdata()
    # process gt
    shape = gt_img.shape
    gt_img_new = np.copy(gt_img)
    for i in range(1, gt_img.shape[2]-1):
        gt_img_new[:,:,i] = (gt_img[:,:,i-1] + gt_img[:,:,i+1]) / 2
    nb.save(nb.Nifti1Image(gt_img_new, nb.load(gt_file).affine), os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56_1/gt_img_avg_slice.nii.gz'))
    gt_img = np.copy(gt_img_new) if avg_slice else np.copy(gt_img)
    gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

    condition_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56_1/condition_img.nii.gz')
    condition_img = nb.load(condition_file).get_fdata() if avg_slice == False else nb.load(condition_file).get_fdata()[:,:,1:shape[2]-1]
    condition_img_brain = Data_processing.cutoff_intensity(condition_img, cutoff_low=-100, cutoff_high=100)

    noise2noise_file = os.path.join('/mnt/camca_NAS/denoising/models/noise2noise_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch78/pred_img.nii.gz')
    noise2noise_img = nb.load(noise2noise_file).get_fdata() if avg_slice == False else nb.load(noise2noise_file).get_fdata()[:,:,1:shape[2]-1]
    noise2noise_img_brain = Data_processing.cutoff_intensity(noise2noise_img, cutoff_low=-100, cutoff_high=100)

    supervised_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch58_1/pred_img.nii.gz')
    supervised_img = nb.load(supervised_file).get_fdata() if avg_slice == False else nb.load(supervised_file).get_fdata()[:,:,1:shape[2]-1]
    supervised_img_brain = Data_processing.cutoff_intensity(supervised_img, cutoff_low=-100, cutoff_high=100)

    # supervised_avg_file = os.path.join('/mnt/camca_NAS/denoising/models/supervised_DDPM_possion_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch50final/pred_img.nii.gz')
    # supervised_avg_img = nb.load(supervised_avg_file).get_fdata() if avg_slice == False else nb.load(supervised_avg_file).get_fdata()[:,:,1:shape[2]-1]
    # supervised_avg_img_brain = Data_processing.cutoff_intensity(supervised_avg_img, cutoff_low=-100, cutoff_high=100)

    ddpm_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56_1/pred_img.nii.gz')
    ddpm_img = nb.load(ddpm_file).get_fdata() if avg_slice == False else nb.load(ddpm_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_img_brain = Data_processing.cutoff_intensity(ddpm_img, cutoff_low=-100, cutoff_high=100)

    ddpm_beta0_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta0/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch61_1/pred_img.nii.gz')
    ddpm_beta0_img = nb.load(ddpm_beta0_file).get_fdata() if avg_slice == False else nb.load(ddpm_beta0_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_beta0_img_brain = Data_processing.cutoff_intensity(ddpm_beta0_img, cutoff_low=-100, cutoff_high=100)

    ddpm_beta20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta20/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch57_1/pred_img.nii.gz')
    ddpm_beta20_img = nb.load(ddpm_beta20_file).get_fdata() if avg_slice == False else nb.load(ddpm_beta20_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_beta20_img_brain = Data_processing.cutoff_intensity(ddpm_beta20_img, cutoff_low=-100, cutoff_high=100)

    ddpm_avg_10_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56avg/pred_img_scans10.nii.gz')
    ddpm_avg_10_img = nb.load(ddpm_avg_10_file).get_fdata() if avg_slice == False else nb.load(ddpm_avg_10_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_avg_10_img_brain = Data_processing.cutoff_intensity(ddpm_avg_10_img, cutoff_low=-100, cutoff_high=100)

    ddpm_avg_20_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_gaussian_2D_mean_beta10/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch56avg/pred_img_scans20.nii.gz')
    ddpm_avg_20_img = nb.load(ddpm_avg_20_file).get_fdata() if avg_slice == False else nb.load(ddpm_avg_20_file).get_fdata()[:,:,1:shape[2]-1]
    ddpm_avg_20_img_brain = Data_processing.cutoff_intensity(ddpm_avg_20_img, cutoff_low=-100, cutoff_high=100)

    # compare the mean value in brain region (crop a ROI in the center)
    x,y = 256,256
    gt_img_brain_ROI = np.clip(gt_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    condition_img_brain_ROI = np.clip(condition_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    noise2noise_img_brain_ROI = np.clip(noise2noise_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    supervised_img_brain_ROI = np.clip(supervised_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    ddpm_img_brain_ROI = np.clip(ddpm_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    ddpm_beta0_img_brain_ROI = np.clip(ddpm_beta0_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    ddpm_beta20_img_brain_ROI = np.clip(ddpm_beta20_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    ddpm_avg_10_img_brain_ROI = np.clip(ddpm_avg_10_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    ddpm_avg_20_img_brain_ROI = np.clip(ddpm_avg_20_img_brain[x-50: x+50, y-50: y+50, 20:40],0,100)
    mean_gt, mean_condition, mean_n2n, mean_supervised, mean_ddpm, mean_ddpm_beta0, mean_ddpm_beta20, mean_ddpm_avg_10, mean_ddpm_avg_20 = np.mean(gt_img_brain_ROI), np.mean(condition_img_brain_ROI), np.mean(noise2noise_img_brain_ROI), np.mean(supervised_img_brain_ROI), np.mean(ddpm_img_brain_ROI), np.mean(ddpm_beta0_img_brain_ROI), np.mean(ddpm_beta20_img_brain_ROI), np.mean(ddpm_avg_10_img_brain_ROI), np.mean(ddpm_avg_20_img_brain_ROI)
    results_mean.append([patient_id, patient_subid, random_n, mean_gt, mean_condition, mean_n2n, mean_supervised, mean_ddpm, mean_ddpm_beta0, mean_ddpm_beta20, mean_ddpm_avg_10, mean_ddpm_avg_20])
    df_mean = pd.DataFrame(results_mean, columns = ['patient_id', 'patient_subid', 'random_n',
    'mean_gt', 'mean_condition', 'mean_n2n', 'mean_supervised', 'mean_ddpm', 'mean_ddpm_beta0', 'mean_ddpm_beta20', 'mean_ddpm_avg_10', 'mean_ddpm_avg_20'])
    file_name_mean = 'mean_measurements.xlsx' 
    df_mean.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name_mean), index = False)


    # compare brain region
    # define eroded mask
    mask = np.zeros(gt_img_brain.shape, dtype=bool)
    mask[(gt_img_brain>0) & (gt_img_brain < 100)] = 1
    print(mask.shape)
    structure = np.ones((6,6))
    mask_eroded = np.zeros_like(mask, dtype=bool)
    for i in range(mask.shape[2]):
        mask_eroded[:, :, i] = binary_erosion(mask[:, :, i], structure=structure, iterations=1)

    mae_brain_motion, _, rmse_brain_motion, _, ssim_brain_motion,psnr_brain_motion = ff.compare(condition_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_n2n, _, rmse_brain_n2n, _, ssim_brain_n2n,psnr_brain_n2n = ff.compare(noise2noise_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_supervised, _, rmse_brain_supervised, _, ssim_brain_supervised,psnr_brain_supervised = ff.compare(supervised_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)

    mae_brain_ddpm, _, rmse_brain_ddpm, _, ssim_brain_ddpm,psnr_brain_ddpm = ff.compare(ddpm_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_beta0, _, rmse_brain_ddpm_beta0, _, ssim_brain_ddpm_beta0,psnr_brain_ddpm_beta0 = ff.compare(ddpm_beta0_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_beta20, _, rmse_brain_ddpm_beta20, _, ssim_brain_ddpm_beta20,psnr_brain_ddpm_beta20 = ff.compare(ddpm_beta20_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_avg_10, _, rmse_brain_ddpm_avg_10, _, ssim_brain_ddpm_avg_10,psnr_brain_ddpm_avg_10 = ff.compare(ddpm_avg_10_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)
    mae_brain_ddpm_avg_20, _, rmse_brain_ddpm_avg_20, _, ssim_brain_ddpm_avg_20,psnr_brain_ddpm_avg_20 = ff.compare(ddpm_avg_20_img_brain[mask_eroded==1], gt_img_brain[mask_eroded==1], cutoff_low = 0, cutoff_high = 100)

    print('motion:', mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion)
    print('n2n:', mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n)
    print('supervised:', mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised)
    print('ddpm:', mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm)
    print('ddpm_beta0:', mae_brain_ddpm_beta0, rmse_brain_ddpm_beta0, ssim_brain_ddpm_beta0, psnr_brain_ddpm_beta0)
    print('ddpm_beta20: ', mae_brain_ddpm_beta20, rmse_brain_ddpm_beta20, ssim_brain_ddpm_beta20, psnr_brain_ddpm_beta20)
    print('ddpm_avg_10:', mae_brain_ddpm_avg_10, rmse_brain_ddpm_avg_10, ssim_brain_ddpm_avg_10, psnr_brain_ddpm_avg_10)
    print('ddpm_avg_20:', mae_brain_ddpm_avg_20, rmse_brain_ddpm_avg_20, ssim_brain_ddpm_avg_20, psnr_brain_ddpm_avg_20)

    # # compare all 
    # mae_motion, _, rmse_motion, _, ssim_motion,psnr_motion = ff.compare(condition_img, gt_img, cutoff_low = -100)
    # mae_n2n, _, rmse_n2n, _, ssim_n2n,psnr_n2n = ff.compare(noise2noise_img, gt_img, cutoff_low = -100)
    # mae_supervised, _, rmse_supervised, _, ssim_supervised,psnr_supervised = ff.compare(supervised_img, gt_img, cutoff_low = -100)
    # mae_ddpm, _, rmse_ddpm, _, ssim_ddpm,psnr_ddpm = ff.compare(ddpm_img, gt_img, cutoff_low = -100)
    # mae_ddpm_avg_10, _, rmse_ddpm_avg_10, _, ssim_ddpm_avg_10,psnr_ddpm_avg_10 = ff.compare(ddpm_avg_10_img, gt_img, cutoff_low = -100)
    # mae_ddpm_avg_20, _, rmse_ddpm_avg_20, _, ssim_ddpm_avg_20,psnr_ddpm_avg_20 = ff.compare(ddpm_avg_20_img, gt_img, cutoff_low = -100)

    # print('all image:')
    # print('motion:', mae_motion, rmse_motion, ssim_motion, psnr_motion)
    # print('n2n:', mae_n2n, rmse_n2n, ssim_n2n, psnr_n2n)
    # print('supervised:', mae_supervised, rmse_supervised, ssim_supervised, psnr_supervised)
    # print('ddpm:', mae_ddpm, rmse_ddpm, ssim_ddpm, psnr_ddpm)
    # print('ddpm_avg_10:', mae_ddpm_avg_10, rmse_ddpm_avg_10, ssim_ddpm_avg_10, psnr_ddpm_avg_10)
    # print('ddpm_avg_20:', mae_ddpm_avg_20, rmse_ddpm_avg_20, ssim_ddpm_avg_20, psnr_ddpm_avg_20)

    # calculate lpips in brain
    lpips_brain_motion = compute_lpips_3d(condition_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_n2n = compute_lpips_3d(noise2noise_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_supervised = compute_lpips_3d(supervised_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_ddpm = compute_lpips_3d(ddpm_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_ddpm_beta0 = compute_lpips_3d(ddpm_beta0_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_ddpm_beta20 = compute_lpips_3d(ddpm_beta20_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_ddpm_avg_10 = compute_lpips_3d(ddpm_avg_10_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)
    lpips_brain_ddpm_avg_20 = compute_lpips_3d(ddpm_avg_20_img_brain, gt_img_brain, max_val = 100, min_val = 0, mask = mask_eroded)

    print('lpips: ')
    print('motion:', lpips_brain_motion)
    print('n2n:', lpips_brain_n2n)
    print('supervised:', lpips_brain_supervised)
    print('ddpm:', lpips_brain_ddpm)
    print('ddpm_beta0:', lpips_brain_ddpm_beta0)
    print('ddpm_beta20:', lpips_brain_ddpm_beta20)
    print('ddpm_avg_10:', lpips_brain_ddpm_avg_10)
    print('ddpm_avg_20:', lpips_brain_ddpm_avg_20)

    results.append([patient_id, patient_subid, random_n, 
    mae_brain_motion, rmse_brain_motion, ssim_brain_motion, psnr_brain_motion, 
    mae_brain_n2n, rmse_brain_n2n, ssim_brain_n2n, psnr_brain_n2n,
    mae_brain_supervised, rmse_brain_supervised, ssim_brain_supervised, psnr_brain_supervised, 
    mae_brain_ddpm, rmse_brain_ddpm, ssim_brain_ddpm, psnr_brain_ddpm, 
    mae_brain_ddpm_beta0, rmse_brain_ddpm_beta0, ssim_brain_ddpm_beta0, psnr_brain_ddpm_beta0,
    mae_brain_ddpm_beta20, rmse_brain_ddpm_beta20, ssim_brain_ddpm_beta20, psnr_brain_ddpm_beta20,
    mae_brain_ddpm_avg_10, rmse_brain_ddpm_avg_10, ssim_brain_ddpm_avg_10, psnr_brain_ddpm_avg_10, 
    mae_brain_ddpm_avg_20, rmse_brain_ddpm_avg_20, ssim_brain_ddpm_avg_20, psnr_brain_ddpm_avg_20, 
    lpips_brain_motion, lpips_brain_n2n, lpips_brain_supervised, lpips_brain_ddpm, lpips_brain_ddpm_beta0,lpips_brain_ddpm_beta20, lpips_brain_ddpm_avg_10, lpips_brain_ddpm_avg_20])
    

    df = pd.DataFrame(results, columns = ['patient_id', 'patient_subid', 'random_n', 
    'mae_brain_motion', 'rmse_brain_motion', 'ssim_brain_motion', 'psnr_brain_motion', 
    'mae_brain_n2n', 'rmse_brain_n2n', 'ssim_brain_n2n', 'psnr_brain_n2n',
    'mae_brain_supervised', 'rmse_brain_supervised', 'ssim_brain_supervised', 'psnr_brain_supervised',
    'mae_brain_ddpm', 'rmse_brain_ddpm', 'ssim_brain_ddpm', 'psnr_brain_ddpm',
    'mae_brain_ddpm_beta0', 'rmse_brain_ddpm_beta0', 'ssim_brain_ddpm_beta0', 'psnr_brain_ddpm_beta0',
    'mae_brain_ddpm_beta20', 'rmse_brain_ddpm_beta20', 'ssim_brain_ddpm_beta20', 'psnr_brain_ddpm_beta20',
    'mae_brain_ddpm_avg_10', 'rmse_brain_ddpm_avg_10', 'ssim_brain_ddpm_avg_10', 'psnr_brain_ddpm_avg_10', 
    'mae_brain_ddpm_avg_20', 'rmse_brain_ddpm_avg_20', 'ssim_brain_ddpm_avg_20', 'psnr_brain_ddpm_avg_20',
    'lpips_brain_motion', 'lpips_brain_n2n', 'lpips_brain_supervised', 'lpips_brain_ddpm', 'lpips_brain_ddpm_beta0','lpips_brain_ddpm_beta20' ,'lpips_brain_ddpm_avg_10', 'lpips_brain_ddpm_avg_20'])
    file_name = 'quantitative_results.xlsx' if avg_slice == False else 'quantitative_results_avg_slice.xlsx'
    df.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)


# results = []
# avg_slice = False
# results = []
# for i in range(0,n.shape[0]):
#     patient_id = patient_id_list[n[i]]
#     patient_subid = patient_subid_list[n[i]]
#     random_n = random_num_list[n[i]]
#     print(patient_id, patient_subid, random_n)
#     r = [patient_id, patient_subid, random_n]

#     gt_file = os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73_1/gt_img.nii.gz')
#     gt_img = nb.load(gt_file).get_fdata()
#     # process gt
#     shape = gt_img.shape
#     gt_img_new = np.zeros((gt_img.shape[0], gt_img.shape[1], gt_img.shape[2]-2))
#     for i in range(1, gt_img.shape[2]-1):
#         gt_img_new[:,:,i-1] = np.mean(gt_img[:,:,i-1:i+2], axis = 2)
#     gt_img = np.copy(gt_img_new) if avg_slice else np.copy(gt_img)
#     gt_img_brain = Data_processing.cutoff_intensity(gt_img, cutoff_low=-100, cutoff_high=100)

#     ddpm_avg_files = ff.sort_timeframe(ff.find_all_target_files(['pred_img_scans*'], os.path.join('/mnt/camca_NAS/denoising/models/unsupervised_DDPM_gaussian_2D/pred_images', patient_id, patient_subid,'random_'+str(random_n), 'epoch73avg')),2,'s','.')
#     print(ddpm_avg_files)
    
#     for k in range(0,len(ddpm_avg_files)):
#         ddpm_avg_img = nb.load(ddpm_avg_files[k]).get_fdata() if avg_slice == False else nb.load(ddpm_avg_files[k]).get_fdata()[:,:,1:shape[2]-1]
#         ddpm_avg_img_brain = Data_processing.cutoff_intensity(ddpm_avg_img, cutoff_low=-100, cutoff_high=100)
#         mae_brain_ddpm_avg, _, rmse_brain_ddpm_avg, _, ssim_brain_ddpm_avg,psnr_brain_ddpm_avg = ff.compare(ddpm_avg_img_brain, gt_img_brain, cutoff_low = 0, cutoff_high = 100)
#         lpips_brain_ddpm_avg = compute_lpips_3d(ddpm_avg_img_brain, gt_img_brain, max_val = 100, min_val = 0)
#         print('when avg num:', k+1, ' mae:', mae_brain_ddpm_avg, ' rmse:', rmse_brain_ddpm_avg, ' ssim:', ssim_brain_ddpm_avg, ' lpips:', lpips_brain_ddpm_avg)
#         r += [mae_brain_ddpm_avg, rmse_brain_ddpm_avg, ssim_brain_ddpm_avg,  lpips_brain_ddpm_avg]

#     results.append(r)

#     columns = ['patient_id', 'patient_subid', 'random_n']
#     for k in range(0,len(ddpm_avg_files)):
#         columns += ['mae_scan'+str(k+1), 'rmse_scan'+str(k+1), 'ssim_scan'+str(k+1), 'lpips_scan'+str(k+1)]
#     df = pd.DataFrame(results, columns = columns)
#     file_name = 'quantitative_results_multiple_scans.xlsx' 
#     df.to_excel(os.path.join('/mnt/camca_NAS/denoising/models', file_name), index = False)

    

   