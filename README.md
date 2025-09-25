# De-noising thin-slice brain CT using diffusion model without clean training data
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on a conference paper submitted to SPIE: <br />
*Noise2Noise Diffusion Model for Thin-Slice Brain CT Denoising without Clean Training Data*<br />
Authors: Zhennong Chen, Siyeop Yoon, Matthew Tivan, Junyoung Park, Quanzheng Li, Dufan Wu<br />

**Citation**: TBD

## Description
will fill in later


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker```. The docker image can be built by ```./docker_build.sh```, after that the docker container can be built by ```./docker_run.sh```. The installed packages can be referred to ```dockerfile``` and ```requirements.txt``` <br />
- You'll need  ```docker/docker_tensorflow``` for step 1 and ```docker/docker_torch``` for the rest steps<br />

### Data Preparation (we have examples available)
You should prepare three things to train the model. Please refer to the `example_data` folder for guidance:

- **fixedCT**  
   - We first re-sampled the original fixedCT slice thickness to 5mm using averaging (which we can consider as low-noise). then we resampled 5mm to 0.625mm to serve as thin-slice CT (low noise as well)
   - All files should be placed in a folder named:  
     `fixedCT`.

- **Simulations**  
   - for each fixed CT case, we run two kinds of simulation: **type 1** refers to poisson noise + hann filter, **type 2** refers to gaussian noise + soft tissue kernel. 

- **A patient list** that enumerates all your cases.  
   - To understand the expected format, please refer to the file:  
     `example_data/Patient_lists`.

- Please refer ```example_data``` folder for examples.


### Experiments
Note we train the diffusion model in two ways:
1. Supervised: we train the model with paired noisy data and low-noise data. Here we use the noisy data with **type 1** noise.
2. Unsupervised (our proposed method): we train the model with only noisy data. Here we use the noisy data with **type 2** noise.<br />
In the testing/inference, we will apply our model to noisy data with **type 2** noise, where we anticipate supervised method will suffer from domain shift error while ours will not.

we have design our study into 5 steps, with each step having its own jupyter notebook.<br /> 
**step1: noise simulation**: use ```step1_simulation.ipynb```. It can simulate both type 1 and 2 noise <br /> 

**step2: split dataset**: use ```step2_split_dataset.ipynb```, it splits data into 6 batches where the first 5 are used for training and validation while the last one is for testing.<br /> 


**step3: train diffusion model**: use ```step3_train_model.ipynb```. It enables the training using supervised or unsupervised ways <br /> 

**step4: Inference**: use ```step4_inference.ipynb```. It first does the inference/sampling for 10/20 times, then take the average to generate the final output <br /> 

**step5: quantitative**: use ```step5_quantitative.ipynb```, need to be corrected, please ignore. <br /> 

### Additional guidelines 
Please contact chenzhennong@gmail.com for any further questions.



