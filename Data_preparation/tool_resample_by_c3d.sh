#!/usr/bin/env bash
# run in docker 

main_path="/mnt/mount_zc_NAS/motion_correction/data/raw_data"
# Get a list of patients.
patients=(${main_path}/nii-images/thin_slice/MO101701M000001/MO001A000001)

img_folder="img-nii-0.625"

for p in ${patients[*]};
do

# Print the current patient.
 
  patient_subid=$(basename ${p})
  patient_id=$(basename $(dirname ${p}))
  
  echo ${p}

  # assert whether nii image exists
  if ! [ -d ${p}/${img_folder} ] || ! [ "$(ls -A  ${p}/${img_folder})" ];then
    echo "no image"
    continue
  fi

  # set output folder
  o_dir=${p}/img-nii-try
  # echo ${o_dir}
  mkdir -p ${o_dir}


  IMGS=(${p}/${img_folder}/*.nii.gz)

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
  #echo ${IMGS[${i}]}
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      c3d ${i_file} -interpolation Cubic -resample-mm 1x1x2.5mm -o ${o_file}
    fi   
  done
done


