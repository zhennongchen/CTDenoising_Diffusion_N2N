# dataset classes

import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Diffusion_denoising_thin_slice.Data_processing as Data_processing
import Diffusion_denoising_thin_slice.functions_collection as ff

# histogram equalization pre-saved load
bins = np.load('/mnt/camca_NAS/denoising/Data/histogram_equalization/bins.npy')
bins_mapped = np.load('/mnt/camca_NAS/denoising/Data/histogram_equalization/bins_mapped.npy')

# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


class Dataset_2D(Dataset):
    def __init__(
        self,
        img_list,
        condition_list,
        image_size,

        num_slices_per_image,
        random_pick_slice,
        slice_range, # None or [a,b]

        supervision, # supervised or unsupervised
        target, # mean or current

        histogram_equalization,
        background_cutoff, 
        maximum_cutoff,
        normalize_factor,

        num_patches_per_slice = None,
        patch_size = None,

        shuffle = False,
        augment = False,
        augment_frequency = 0,
    ):
        super().__init__()
        self.img_list = img_list
        self.condition_list = condition_list
        self.image_size = image_size
        self.num_slices_per_image = num_slices_per_image
        self.random_pick_slice = random_pick_slice
        self.slice_range = slice_range
        self.num_patches_per_slice = num_patches_per_slice
        self.patch_size = patch_size

        self.supervision = supervision
        self.target = target

        self.histogram_equalization = histogram_equalization
        self.background_cutoff = background_cutoff
        self.maximum_cutoff = maximum_cutoff
        self.normalize_factor = normalize_factor
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(img_list)

        self.index_array = self.generate_index_array()
        self.current_x0_file = None
        self.current_x0_data = None
        self.current_condition_file = None
        self.current_condition_data = None
       

    def generate_index_array(self):
        np.random.seed()
        index_array = []; index_array_patches = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            s_list = np.arange(self.num_slices_per_image)
            for s in s_list:
                index_array.append([f, s])
                if self.num_patches_per_slice != None:
                    patch_list = np.arange(self.num_patches_per_slice)
                    for p in patch_list:
                        index_array_patches.append([f, s,p])
        if self.num_patches_per_slice != None:
            return index_array_patches
        else:
            return index_array

    def __len__(self):
        if self.num_patches_per_slice != None:
            return self.num_files * self.num_slices_per_image * self.num_patches_per_slice
        else:
            return self.num_files * self.num_slices_per_image
    

    def load_file(self, filename):
        ii = nb.load(filename).get_fdata()
    
        # do histogram equalization first
        if self.histogram_equalization == True: 
            ii = Data_processing.apply_transfer_to_img(ii, bins, bins_mapped)
        # cutoff and normalization
        ii = Data_processing.cutoff_intensity(ii,cutoff_low = self.background_cutoff, cutoff_high = self.maximum_cutoff)
        ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)
        ii = Data_processing.crop_or_pad(ii, [self.image_size[0], self.image_size[1], ii.shape[2]], value= np.min(ii))

        return ii
        
    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        if self.num_patches_per_slice != None:
            f,s,p = self.index_array[index]
        else:
            f,s = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        x0_filename = self.img_list[f]
        # print('x0 filename is: ', x0_filename, ' while current x0 file is: ', self.current_x0_file)
        condition_file = self.condition_list[f]
        # print('condition file is: ', condition_file, ' while current condition file is: ', self.current_condition_file)

        if self.supervision == 'supervised': # in unsupervised case, we do not need to load the x0 file since we don't have clean image
            # print('we have x0 since we have clean image')
            if x0_filename != self.current_x0_file:
                x0_img = self.load_file(x0_filename)
                # print('load: ',x0_filename)
                self.current_x0_file = x0_filename
                self.current_x0_data = np.copy(x0_img)

        if condition_file != self.current_condition_file:
            # print('it is a new case, load the file')
            condition_img = self.load_file(condition_file)
            self.current_condition_file = condition_file
            self.current_condition_data = np.copy(condition_img)

            if self.supervision == 'unsupervised':
                self.current_x0_data = np.copy(self.current_condition_data)

            # define a list of random slice numbers
            if self.slice_range == None:
                total_slice_range = [0,self.current_condition_data.shape[2]] if self.supervision == 'supervised' else [0 + 1,self.current_condition_data.shape[2]-1]
            else:
                total_slice_range = self.slice_range
            # print('in this condition case, total slice range is: ', total_slice_range)
            if self.random_pick_slice == False:
                self.slice_index_list = np.arange(total_slice_range[0], total_slice_range[1])
                self.slice_index_list = self.slice_index_list[:self.num_slices_per_image]
            else:
                self.slice_index_list = np.random.permutation(np.arange(total_slice_range[0], total_slice_range[1]))[:self.num_slices_per_image]
            # print('in this condition case, slice index list is: ', self.slice_index_list)

        # pick the slice
        # print('pick the slice: ', self.slice_index_list[s])
        s = self.slice_index_list[s]

        # pick the patch:
        if self.num_patches_per_slice != None:
            x_shape, y_shape = self.current_condition_data.shape[0], self.current_condition_data.shape[1]
            random_origin_x, random_origin_y = random.randint(0, x_shape - self.patch_size[0]), random.randint(0, y_shape - self.patch_size[1])
            # print('x range is: ', random_origin_x, random_origin_x + self.patch_size[0], ' and y range is: ', random_origin_y, random_origin_y + self.patch_size[1])

        # target image
        x0_image_data = np.copy(self.current_x0_data)[:,:,s] # we have clean data
        if self.target == 'mean':
            x0_image_data = (self.current_x0_data[:,:,s-1] + self.current_x0_data[:,:,s+1]) / 2
        # crop the patch
        if self.num_patches_per_slice != None:
            x0_image_data = x0_image_data[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
        
        # condition image
        if self.supervision == 'supervised':
            condition_image_data = np.copy(self.current_condition_data)[:,:,s]
            if self.num_patches_per_slice != None:
                condition_image_data = condition_image_data[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
        elif self.supervision == 'unsupervised':
            if self.target == 'current': # use neighboring slices as condition
                condition_image_data1 = np.copy(self.current_condition_data)[:,:,s-1]
                condition_image_data2 = np.copy(self.current_condition_data)[:,:,s+1]
                if self.num_patches_per_slice != None:
                    condition_image_data1 = condition_image_data1[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
                    condition_image_data2 = condition_image_data2[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
                condition_image_data = np.stack([condition_image_data1, condition_image_data2], axis = -1)
            elif self.target == 'mean': # use current slice as condition
                condition_image_data = np.copy(self.current_condition_data)[:,:,s]
                if self.num_patches_per_slice != None:
                    condition_image_data = condition_image_data[random_origin_x:random_origin_x + self.patch_size[0], random_origin_y:random_origin_y + self.patch_size[1]]
            # print('shape of condition image data: ', condition_image_data.shape)

        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                x0_image_data, z_rotate_degree = random_rotate(x0_image_data,  order = 1)
                x0_image_data, x_translate, y_translate = random_translate(x0_image_data)
                condition_image_data, _ = random_rotate(condition_image_data, z_rotate_degree = z_rotate_degree, order = 1)
                condition_image_data, _, _ = random_translate(condition_image_data, x_translate = x_translate, y_translate = y_translate)
                # print('augment : z_rotate_degree, x_translate, y_translate: ', z_rotate_degree, x_translate, y_translate)
        
            
        x0_image_data = torch.from_numpy(x0_image_data).unsqueeze(0).float()
        if self.supervision == 'supervised':
            condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()
        elif self.supervision == 'unsupervised':
            if self.target == 'current':
                condition_image_data = np.transpose(condition_image_data, (2,0,1))
                condition_image_data = torch.from_numpy(condition_image_data).float()
            elif self.target == 'mean':
                condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()

        # print('shape of x0 image data: ', x0_image_data.shape, ' and condition image data: ', condition_image_data.shape)
        return x0_image_data, condition_image_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
    
