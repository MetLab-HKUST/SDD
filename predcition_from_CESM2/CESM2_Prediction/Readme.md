## Scripts applying the deep learning model to CESM2 data

The "generate*ncl" scripts computes the daily mean circulation patters in a predefined domain. The domain is defined around the targeted prediction region (South China in our study) and has sufficient margins so that for each grid cell in the targedted region we can extract a 48 degree by 48 degree square centerd at that grid cell.

The "DL_Predict_CESM2_SSP585.py" script takes in the CESM2 circulation data and apply the trained deep learning model to make prediction about subgrid maximum daily precipitation. Note that unlike the training stage, we did not put the CESM2 circulation data onto 48 degree by 48 degree squares before running this prediction script. Instead, this remapping happens in this python script while running. Doing it this way saves storage space, but is optional. Applying the deep learning model is very quick so we did not bother to save the deep learning datasets in advance.

The MATLAB script pots the prediction results (see the TIFF file in this folder).
