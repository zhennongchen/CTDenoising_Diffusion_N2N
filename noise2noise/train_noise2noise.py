import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_denoising_thin_slice.noise2noise.model as noise2noise
import Diffusion_denoising_thin_slice.functions_collection as ff
import Diffusion_denoising_thin_slice.Build_lists.Build_list as Build_list
import Diffusion_denoising_thin_slice.noise2noise.Generator as Generator

#######################
trial_name = 'noise2noise_2D'
pre_trained_model = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models', 'model-13.pt')
start_step = 13
image_size = [512,512]
num_patches_per_slice = 2
patch_size = [128,128]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
#######################
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/denoising/Patient_lists/fixedCT_static_simulation_train_test_gaussian_local.xlsx'))
_,_,_,_, condition_list_train, x0_list_train = build_sheet.__build__(batch_list = [0,1,2,3]) 
# x0_list_train = x0_list_train[0:1]; condition_list_train = condition_list_train[0:1]
 
# define val
_,_,_,_, condition_list_val, x0_list_val = build_sheet.__build__(batch_list = [4])
# x0_list_val = x0_list_val[0:1]; condition_list_val = condition_list_val[0:1]

print('train:', x0_list_train.shape, condition_list_train.shape, 'val:', x0_list_val.shape, condition_list_val.shape)
print(x0_list_train[0:5], condition_list_train[0:5], x0_list_val[0:5], condition_list_val[0:5])


# build model
model = noise2noise.Unet2D(
    init_dim = 16,
    channels = 2, 
    out_dim = 1,
    dim_mults = (2,4,8,16),
    full_attn = (None,None, False, True),
    act = 'ReLU',
)

# build generator
generator_train = Generator.Dataset_2D(
        img_list = condition_list_train, 
        image_size = image_size,

        num_slices_per_image = 50,
        random_pick_slice = True,
        slice_range = None,

        num_patches_per_slice = num_patches_per_slice,
        patch_size = patch_size,

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,

        shuffle = True,
        augment = True,
        augment_frequency = 0.5,)

generator_val = Generator.Dataset_2D(
        img_list = condition_list_val,
        image_size = image_size,

        num_slices_per_image = 20,
        random_pick_slice = False,
        slice_range = [50,70],

        num_patches_per_slice = 1,
        patch_size = [512,512],

        histogram_equalization = histogram_equalization,
        background_cutoff = background_cutoff,
        maximum_cutoff = maximum_cutoff,
        normalize_factor = normalize_factor,)


# train
trainer = noise2noise.Trainer(
    model= model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 25,

    train_num_steps = 10000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/denoising/models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 200, 
    save_models_every = 1,
    validation_every = 1,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)