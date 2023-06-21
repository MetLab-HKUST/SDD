## Scripts for traning and testing the deep learning model 

### 1. Prepare_Dataset4DL.py
Data created by the preprocessing step save each variable in a separate file. This script merge, shuffle, and concatenate them to create the dataset usable by the deep learning model training script. Note that data are normalzied (scaled) by the mean and standard deviation of each variable at each pressure level. The scaling parameters are saved for later use. Different variables and different levels are treated as the 'channel' dimension in the datasets. Data are divided into training, testing, and validation sets.

### 2. Train_DL_Parallel.py and Train_DL_SingleGPU.py
These are the scripts for training the deep learning model, one using a single GPU and the other using distributed training on multiple GPUs. Because the whole dataset is too big for my GPU memory, I load the dataset into CPU memory first and then feed subsets to GPU memory in a loop to complete the training. If your work demands even more memory than your CPU memory, you need to save the subsets on to disks and load those files in a loop. My scripts save the DL model with lowest validation loss.

### 3. Test_DL_Prediction.py
Run the saved deep learning model on testing dataset and compute the loss.

### 4. Plot_DL_Tests.m
Plot a scatter plot and a few metrics to measure the performance of the trained model (see the JPG file in this folder)

