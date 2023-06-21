# SDD
Codes for smart dynamical downscaling (SDD)

This repository has code for SDD. The current code trains deep-learning (DL) models to predict subgrid precipitation maximum based on coarse-resolution global climate model simulations. Then based on the DL model prediction, we select those time slices likely to produce extreme events and conduct dynamical downscaling, e.g., with WRF. The dynamical downscaling is supposed to have a very high resolution, so running it for decades is too computationally demanding. Instead, we conduct targeted high-resolution simulations. 

The workflow is usually in the following order:
## 1. Preprocessing
Prepare precipitation and circulation data (e.g., ERA5 reanalysis) for deep learning.
## 2. Deep-learning model training
Create training datasets and train the DL model.
## 3. Make DL model predictions
Apply the trained DL model to climate model data (e.g., CESM2) to predict the precipitation in the targeted domain.
## 4. Prepared initial and boundary conditions
Based on the DL model prediction, extract climate model data on selected dates and generate initial and boundary conditions for dynamical downscaling with WRF.
## 5. Run dynamical downscaling simulations
Run WRF with the initial and boundary conditions for each case.

This repository provides code and scripts for steps 1-4. Each folder has its own README to explain the purpose of code. 
