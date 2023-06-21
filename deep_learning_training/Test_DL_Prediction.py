# Training the RxNet for rainfall statistics regression

# This script read reanalysis and rainfall momentum (mean and variance) and train the
# RxNet for predicting the mean and variance of precipitation in a 1 degree by 1 degree
# grid cell, which contains 4x4 subgrid cells.


# Import libraries
import tensorflow as tf
# import tensorflow_probability as tfp
import netCDF4 as nc4
import numpy as np

# Load and prepare the dataset
dsFile = nc4.Dataset("DL_ERA5_Testing_Scaled.nc")
testLS0 = dsFile.variables["testingLS"]
testMOM0 = dsFile.variables["testingPrecip"]
testLS = np.copy(testLS0)
testMOM = np.copy(testMOM0[:, 2]).reshape((testLS.shape[0], 1))


RxNet = tf.keras.models.load_model('RxNet_Max_Only.h5')
RxNet.compile()


rec_max = np.empty((testLS.shape[0], 1)).astype("float32")
batch_size = 256
ts = 0
te = batch_size
# Set up DL dataset
testLS_dataset = tf.data.Dataset.from_tensor_slices(testLS).batch(batch_size)
for ls_x in testLS_dataset:
    rec_max[ts:te, 0] = np.squeeze(RxNet(ls_x))
    ts = ts + batch_size
    te = te + batch_size
    if te > testLS.shape[0]:
        te = testLS.shape[0]


# save reconstructed dataset for visualization
itime = np.copy(dsFile.variables["time"])
ncfile = nc4.Dataset(
    "./rxnet_rec_max_only_test.nc", mode="w", format="NETCDF4_CLASSIC"
)

max_dim = ncfile.createDimension("max", 1)
time_dim = ncfile.createDimension("time", None)

maxDim = ncfile.createVariable("max", np.float32, ("max",))
maxDim.units = " "
maxDim.long_name = "maximum"

time = ncfile.createVariable("time", np.float32, ("time",))
time.units = "day"
time.long_name = "dummy time dimension"

rec = ncfile.createVariable(
    "recMax", np.float32, ("time", "max"), fill_value=1.0e36
)
rec.units = "mm/day"
rec.long_name = "rxnet prediction of maximum daily rainfall"

obs = ncfile.createVariable(
    "obsMax", np.float32, ("time", "max"), fill_value=1.0e36
)
obs.units = "mm/day"
obs.long_name = "observed maximum daily rainfall"

maxDim[:] = np.array([1])
time[:] = itime
rec[:, :] = rec_max
obs[:, :] = testMOM

ncfile.close()

testLoss = np.mean((rec_max[:, 0] - testMOM[:, 0])**2)
print("* Testing Loss for mean = {:12.6f}".format(testLoss))
# * Testing Loss for mean =    14.009955
