#!/usr/bin/env bash

# run in terminal of your own laptop

##############
## Settings ##
##############

set -o nounset
set -o errexit
set -o pipefail

#shopt -s globstar nullglob

###########
## Logic ##
###########

# define the folder where dcm2niix function is saved, save it in your local laptop
dcm2niix_fld="/Users/zhennongchen/Documents/GitHub/AI_reslice_orthogonal_view/dcm2niix_11-Apr-2019/"

# main_path="/Volumes/Camca/home/ZC/Portable_CT_data"
main_path="/Volumes/Camca/home/ZC/diffusion_ct_motion/examples/real_data/data"
# define patient lists (the directory where you save all the patient data)
PATIENTS=(${main_path}/*/*)

echo ${#PATIENTS[@]}


for p in ${PATIENTS[*]};
do

  echo ${p}
  
  if [ -d ${p} ];
  then

  patient_id=$(basename $(dirname $(dirname ${p})))
  patient_subid=$(basename $(dirname ${p}))

  # output_folder=${main_path}/nii_imgs_202404/motion
  # mkdir -p ${output_folder}/${patient_id}/
  # mkdir -p ${output_folder}/${patient_id}/${patient_subid}/
  # nii_folder=${output_folder}/${patient_id}/${patient_subid}/fixed/
  # original_folder=${main_path}/IRB2022P002233_collected_202404/${patient_id}/${patient_subid}
  # IMGS=(${original_folder}/*)  # Find all the images under this patient ID

  output_folder=${p}
  nii_folder=${output_folder}
  original_folder=${p}
  IMGS=(${original_folder}/fixed_CT_5mm)  # Find all the images under this patient ID


  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
      do

      echo ${IMGS[${i}]}
      
      if [ "$(ls -A ${IMGS[${i}]})" ]; then  # check whether the image folder is empty
        
        # filename='img_'${i}
        filename='fixed_CT_5mm_high-res'
        o_file=${nii_folder}${filename}.nii.gz # define the name of output nii files, the name will be "timeframe.nii.gz"
        echo ${o_file}

        if [ -f ${o_file} ];then
          echo "already done this file"
          continue

        else
        # if dcm2niix doesn't work (error = ignore image), remove -i y
        ${dcm2niix_fld}dcm2niix -i y -m y -b n -o "${nii_folder}" -f "${filename}" -9 -z y "${IMGS[${i}]}"
        fi

      else
        echo "${IMGS[${i}]} is emtpy; Skipping"
        continue
      fi
      
    done

  else
    echo "${p} missing dicom image folder"
    continue
    
  fi

done
