import numpy as np
import glob 
import os
from PIL import Image

# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width
    low = level - width
    # normalize
    unit = (1-0) / (width*2)
    image[image>high] = high
    image[image<low] = low
    new = (image - low) * unit 
    return new


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

# function: write txt file
def txt_writer(save_path,parameters,names):
    t_file = open(save_path,"w+")
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(round(parameters[i][ii],2))+' ')
        t_file.write('\n')
    t_file.close()



# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = (a-np.min(a)) / (np.max(a) - np.min(a))

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


