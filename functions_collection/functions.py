import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
import CTDenoising_Diffusion_N2N.Data_processing as dp
from skimage.metrics import structural_similarity as compare_ssim
# import CTProjector.src.ct_projector.projector.numpy as ct_projector

def create_lowpass_mask(shape, radius):
    H, W, S = shape
    center = np.array([H // 2, W // 2, S // 2])
    y, x, z = np.ogrid[:H, :W, :S]  # 顺序和 shape 对应
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
    mask = dist <= radius
    return mask

def create_2d_lowpass_mask(shape, radius):
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = dist <= radius
    return mask


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


# function: normalize translation control points:
def convert_translation_control_points(t, dim, from_pixel_to_1 = True):
    if from_pixel_to_1 == True: # convert to a space -1 ~ 1
        t = [tt / dim * 2 for tt in t]
    else: # backwards
        t = [tt / 2 * dim for tt in t]
    
    return np.asarray(t)


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


# function: dice
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


# function: erode and dilate
def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    if erode is True:
        img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary



# function: round difference: if abs(a-b)  % 1 <= threshold, then a-b = math.floor(a-b)
def round_diff(pred, gt, threshold):
    # b is ground truth
    A = pred - gt 
    rounded_A = np.where((np.abs(A) % 1 <= threshold) & (A < 0), np.ceil(A), np.where((np.abs(A) % 1 <= threshold) & (A > 0), np.floor(A), A))
    return gt + rounded_A

# # function: read real-scan projection data
# def read_projection_data(
#     input_dir, projector: ct_projector.ct_projector, start_view, end_view, nrot_per_slab, nz_per_slice
# ):
#     min_file_bytes = 1024 * 10  # file size should be at least 10MB

#     if end_view < 0:
#         end_view = len(find_all_target_files(['slab_*.nii.gz'], input_dir)) -1 
       
#     filenames = []
#     for iview in range(start_view, end_view + 1):
#         filename = os.path.join(input_dir, f'slab_{iview}.nii.gz')
#         if not os.path.exists(filename):
#             break
#         if os.path.getsize(filename) < min_file_bytes:
#             break
#         filenames.append(filename)

#     prjs = []
#     print('Reading data from {0} files'.format(len(filenames)), flush=True)
#     for i, filename in enumerate(filenames):
#         print(i, end=',', flush=True)
#         prj = sitk.GetArrayFromImage(sitk.ReadImage(filename))
#         prj = prj.astype(np.float32)
#         # print('shape: ', prj.shape)
#         prjs.append(prj)
#     prjs = np.array(prjs)

#     prjs = prjs.reshape([
#         nrot_per_slab,
#         prjs.shape[0] // nrot_per_slab,
#         prjs.shape[1],
#         prjs.shape[2] // nz_per_slice,
#         nz_per_slice,
#         prjs.shape[3],
#     ])

#     print('prjs shape ',prjs.shape)

#     prjs = np.mean(prjs, axis=(0, 4))

#     print('prjs shape ',prjs.shape)

#     # update projector
#     projector.nv = prjs.shape[2]
#     projector.nz = prjs.shape[2]
#     projector.dv = projector.dv * nz_per_slice
#     projector.dz = projector.dv

#     return prjs, projector

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

    
