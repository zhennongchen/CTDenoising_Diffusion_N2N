
import os
import numpy as np
import pandas as pd
import shutil
import pydicom
import Diffusion_models.functions_collection as ff

list_folder = '/mnt/camca_NAS/Portable_CT_data/Patient_list'
nas_folder = '/mnt/camca_NAS/Portable_CT_data/IRB2022P002233_collected_202404'

# transfer from one nas to another, only transfer thin slice
ct_list = pd.read_excel(os.path.join(list_folder, 'NEW_CT_concise_collected.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
if 'thin_slice_thickness' not in ct_list.columns:
    ct_list['thin_slice_thickness'] = ''
    ct_list['thin_slice_folder'] = ''

for i in range(0,200):#ct_list.shape[0]):
    if ct_list.iloc[i]['Have_CT_collected'] == False:
        continue
    patient_id = ct_list.iloc[i]['Patient_ID']
    patient_subid = ct_list.iloc[i]['Patient_subID']
    ct_type = ct_list.iloc[i]['CT type']
    folder_path = os.path.join(nas_folder, patient_id, patient_subid)
    print(patient_id, patient_subid, folder_path)

    
    # find all data folders
    data_folders = ff.find_all_target_files(['*'], folder_path)
    data_folders = list(set(data_folders))
    
    if len(data_folders) > 0:
        slice_thickness_mm_list = []
        for data_folder in data_folders:
            dicom_files = ff.find_all_target_files(['*.dcm'], data_folder)
            if dicom_files.shape[0] < 20:
                continue
            metadata = pydicom.dcmread(dicom_files[0])
            slicethickness = metadata.SliceThickness
            if slicethickness is not None:
                slice_thickness_mm_list.append(slicethickness)
        # find the thinnest slice thickness
        if len(slice_thickness_mm_list) == 0:
            thin_slice_tickness = ''
            thin_slice_folder = ''
        else:
            slice_thickness_mm_list = np.array(slice_thickness_mm_list)
            thin_slice_tickness = np.min(slice_thickness_mm_list)
            # corresponding folder name
            thin_slice_folder = data_folders[np.argmin(slice_thickness_mm_list)]
    else:
        thin_slice_tickness = ''
        thin_slice_folder = ''
    print(ct_type, thin_slice_tickness, thin_slice_folder)

    ct_list['thin_slice_thickness'].iloc[i] = thin_slice_tickness
    ct_list['thin_slice_folder'].iloc[i] = thin_slice_folder
    ct_list.to_excel(os.path.join(list_folder, 'NEW_CT_concise_collected_thin_slice_info.xlsx'), index=False)

      