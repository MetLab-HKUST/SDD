## Preprocessing Precipitation Data

This folder has NCL scripts to prepare the precipitation data for deep learning. The task was separated into a few steps so that intermediate results can be saved and examined. 

The key script is Step 2. The original precipitation data have dimensions of [time, y, x] and their resolution is 0.25 degree by 0.25 degree. I divided the grid mesh into 4x4 blocks (1 degree by 1 degree), and reshape the precipitation data into two dimension [time*y*x, 4*4], so that each block is one sample. Then I saved the average, min, max, and variance of each block into separate arrays. During this process, the blocks with less than 3 points on land are discarded. 

The deep learning model in our paper only used the max from each block. Other information (min, average, varaince) were saved because we experimented with probabilistic prediction, which was not successful in our work but worth pursuing in the future.
