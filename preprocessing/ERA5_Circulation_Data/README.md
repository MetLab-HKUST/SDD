## Preparing ERA5 circulation data for deep learning

In this part, we first coarsen the ERA5 data to 1 degree by 1 degree grid mesh to match the resolution of GCMs. Then the "preprocess" scripts convert the data into deep learning *samples*. In our algorithm, the precipitation (subgrid maximum) of each day and each grid cell is one sample. Therefore, the circulation pattern in a 48 degree by 48 degree square centered at the precipitation grid cell is the corresponding sample circulation. The 6-hourly data were avareged to yield a daily mean circulation pattern. The "preprocess" scripts complete the interpolation tasks for each sample and store the circulation data into an array of the dimensions [sample, level, lat, lon]. 

Doing the interpolation with NCL is a bit slow so I have run scripts in this folder to submit those tasks to the job queue and let them run in the background. 
