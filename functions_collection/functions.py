import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
import lpips
import torch
import CTDenoising_Diffusion_N2N.Data_processing as dp
from skimage.metrics import structural_similarity as compare_ssim


def pick_random_from_segments(X):
    # Generate the list from 0 to X
    full_list = list(range(X + 1))
    
    # Determine the segment size
    segment_size = len(full_list) // 4

    # Initialize selected numbers
    selected_numbers = []

    # Loop through each segment and randomly pick one number
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 3 else len(full_list)  # Ensure last segment captures all remaining elements
        segment = full_list[start:end]
        selected_numbers.append(random.choice(segment))

    return selected_numbers


# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


# function: comparison error
def compare(a, b,  cutoff_low = 0 ,cutoff_high = 1000000):
    # compare a to b, b is ground truth
    # if a pixel is lower than cutoff (meaning it's background), then it's out of comparison
    c = np.copy(b)
    diff = abs(a-b)
   
    a = a[(c>cutoff_low)& (c < cutoff_high) ].reshape(-1)
    b = b[(c>cutoff_low)& (c < cutoff_high) ].reshape(-1)

    diff = abs(a-b)

    # mean absolute error
    mae = np.mean(abs(a - b)) 

    # mean squared error
    mse = np.mean((a-b)**2) 

    # root mean squared error
    rmse = math.sqrt(mse)

    # relative root mean squared error
    dominator = math.sqrt(np.mean(b ** 2))
    r_rmse = rmse / dominator * 100

    # structural similarity index metric
    cov = np.cov(a,b)[0,1]
    ssim = (2 * np.mean(a) * np.mean(b)) * (2 * cov) / (np.mean(a) ** 2 + np.mean(b) ** 2) / (np.std(a) ** 2 + np.std(b) ** 2)
    # ssim = compare_ssim(a,b)

    # # normalized mean squared error
    # nmse = np.mean((a-b)**2) / mean_square_value

    # # normalized root mean squared error
    # nrmse = rmse / mean_square_value

    # peak signal-to-noise ratio
    if cutoff_high < 1000:
        max_value = cutoff_high
    else:
        max_value = np.max(b)
    psnr = 10 * (math.log10((max_value**2) / mse ))

    return mae, mse, rmse, r_rmse, ssim,psnr


# function: hanning filter
def hann_filter(x, projector):
    x_prime = np.fft.fft(x)
    x_prime = np.fft.fftshift(x_prime)
    hanning_window = np.hanning(projector.nu)
    x_prime_hann = x_prime * hanning_window
    x_inverse_hann = np.fft.ifft(np.fft.ifftshift(x_prime_hann))
    return x_inverse_hann

def apply_hann(prjs, projector):
    prjs_hann = np.zeros_like(prjs)
    for ii in range(0,prjs_hann.shape[0]):
        for jj in range(0, prjs_hann.shape[2]):
            for kk in range(0, prjs_hann.shape[1]):
                prjs_hann[ii,kk,jj,:] = hann_filter(prjs[ii,kk,jj,:], projector)
    return prjs_hann


# function: patch definition:
def patch_definition(img_shape, patch_size, stride, count_for_overlap = False):
    # now assume patch_size is square in x and y, and same dimension as img in z 
    start = 0
    end = img_shape[0] - patch_size

    origin_x_list = np.arange(start, end+1, stride)

    patch_origin_list = [[origin_x_list[i], origin_x_list[j]] for i in range(0,origin_x_list.shape[0]) for j in range(0,origin_x_list.shape[0])]
    
    if count_for_overlap == False:
        return patch_origin_list, 0
    else:
        count = np.zeros(img_shape)
        for origin in patch_origin_list:
            count[origin[0] : (origin[0] + patch_size), origin[1] : (origin[1] + patch_size), :] += 1

        return patch_origin_list, count

# function: randomly sample patch origins
def sample_patch_origins(patch_origins, N, include_original_list = None):
    if isinstance(patch_origins, list) == False:
        patch_origins = patch_origins.tolist()
    origins = np.array(patch_origins)
    x_min,  x_max, y_min, y_max = np.min(origins[:,0]), np.max(origins[:,0]), np.min(origins[:,1]), np.max(origins[:,1])

    # Generate random coordinates
    pixels = [(random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(N)]

    if include_original_list is True:
        pixels = patch_origins + pixels

    return pixels

# function: compute LPIPS for 3D data
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

    
