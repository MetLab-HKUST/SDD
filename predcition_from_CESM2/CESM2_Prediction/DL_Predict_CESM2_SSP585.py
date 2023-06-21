# Training the RxNet for rainfall statistics regression

# This script read reanalysis and rainfall momentum (mean and variance) and train the
# RxNet for predicting the mean and variance of precipitation in a 1 degree by 1 degree
# grid cell, which contains 4x4 subgrid cells.


# Import libraries
import tensorflow as tf
import os
import sys
import netCDF4 as nc4
import numpy as np


# Load circulation data, one by one
dsFile = nc4.Dataset("DL_CESM2_SSP585_Geopotential.nc")
Z0 = dsFile.variables["Z"]
Z = np.copy(Z0)
Zpe = np.transpose(Z, (0, 2, 3, 1))
# (time, lat, lon, level)
del Z, Z0
if np.any(np.isnan(Zpe)):
    print("!!NaN found in Z")

LS = np.copy(Zpe)
del Zpe

dsFile = nc4.Dataset("DL_CESM2_SSP585_Specific_Humidity.nc")
Q0 = dsFile.variables["Q"]
Q = np.copy(Q0)
Qpe = np.transpose(Q, (0, 2, 3, 1))
# (time, lat, lon, level)
del Q, Q0
if np.any(np.isnan(Qpe)):
    print("!!NaN found in Q")

LS = np.concatenate((LS, Qpe), axis=3)
del Qpe

dsFile = nc4.Dataset("DL_CESM2_SSP585_Surface_Geopotential.nc")
Zs0 = dsFile.variables["Zsfc"]
Zs = np.copy(Zs0)
ZsPe = np.transpose(Zs, (0, 2, 3, 1))
# (time, lat, lon, level)
del Zs, Zs0
if np.any(np.isnan(ZsPe)):
    print("!!NaN found in Zsfc")

LS = np.concatenate((LS, ZsPe), axis=3)
del ZsPe

dsFile = nc4.Dataset("DL_CESM2_SSP585_Temperature.nc")
T0 = dsFile.variables["T"]
T = np.copy(T0)
Tpe = np.transpose(T, (0, 2, 3, 1))
# (time, lat, lon, level)
del T, T0
if np.any(np.isnan(Tpe)):
    print("!!NaN found in T")

LS = np.concatenate((LS, Tpe), axis=3)
del Tpe

dsFile = nc4.Dataset("DL_CESM2_SSP585_U_Wind.nc")
U0 = dsFile.variables["U"]
U = np.copy(U0)
Upe = np.transpose(U, (0, 2, 3, 1))
# (time, lat, lon, level)
del U, U0
if np.any(np.isnan(Upe)):
    print("!!NaN found in U")

LS = np.concatenate((LS, Upe), axis=3)
del Upe

dsFile = nc4.Dataset("DL_CESM2_SSP585_V_Wind.nc")
V0 = dsFile.variables["V"]
V = np.copy(V0)
Vpe = np.transpose(V, (0, 2, 3, 1))
# (time, lat, lon, level)
del V, V0
if np.any(np.isnan(Vpe)):
    print("!!NaN found in V")

LS = np.concatenate((LS, Vpe), axis=3)
del Vpe

# scale the large-scale (LS) circulation data with ERA5 scaling parameters
dsFile = nc4.Dataset("DL_ERA5_Scaling_Parameters.nc")
LSmean = np.copy(dsFile.variables["LSmean"])
LSstd = np.copy(dsFile.variables["LSstd"])

# create array to save precipitation prediction
LSsize = LS.shape
print(f"-- shape of LS: %s" % (LSsize,))
precip = np.ones((LSsize[0], 5, 10)) * -999.9

# read in land mask
dsFile = nc4.Dataset("SouthChina_LandMask_4DL.nc")
isLand = np.copy(dsFile.variables["land_mask"])

# load trained RxNet
RxNet = tf.keras.models.load_model("RxNet_Max_Only.h5")
RxNet.compile()

# loop through all times to predict rainfall
LSdata = np.zeros((28, 48, 48, 31))

for k in np.arange(LSsize[0]):
    if (k % 100) == 0:
        print("progress ")
        print("-- progress:  {:5.2f} %".format(k / LSsize[0] * 100.0))

    # Set up DL dataset
    iSample = 0
    for j in np.arange(5):
        for i in np.arange(10):
            if isLand[j, i] > 0.5:
                LSdata[iSample, :, :, :] = LS[k, j : j + 48, i : i + 48, :]
                iSample += 1

    LSdata = (LSdata - LSmean) / LSstd
    LS_dataset = tf.data.Dataset.from_tensor_slices(LSdata).batch(28)

    # Predict
    iCheck = 0
    for LS_obs in LS_dataset:
        pred_max = np.squeeze(RxNet(LS_obs))
        iCheck += 1

    if iCheck > 1:
        sys.exit("!! Batch size setup problem ... ")

    # Store results
    iSample = 0
    for j in np.arange(5):
        for i in np.arange(10):
            if isLand[j, i] > 0.5:
                precip[k, j, i] = pred_max[iSample]
                iSample += 1


# save predicted
dsFile = nc4.Dataset("DL_CESM2_SSP585_Geopotential.nc")
itime = np.copy(dsFile.variables["time"])

filename = "RxNet_Prediction_CESM2_SSP585.nc"
if os.path.exists(filename):
    os.remove(filename)

ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 10)
ncfile.createDimension("lat", 5)

time_nc = ncfile.createVariable("time", np.float32, ("time",))
time_nc.long_name = "time"
time_nc.standard_name = "time"
time_nc.calendar = "365_day"
time_nc.units = "days since 0001-01-01 00:00:00"

lat_nc = ncfile.createVariable("lat", np.float32, ("lat",))
lat_nc.long_name = "lat"
lon_nc = ncfile.createVariable("lon", np.float32, ("lon",))
lon_nc.long_name = "lon"

precip_nc = ncfile.createVariable(
    "precip", np.float32, ("time", "lat", "lon"), fill_value=-999.9
)
precip_nc.long_name = (
    "RxNet prediction of rainfall maximum for 0.25 deg x 0.25 deg subgrid cells"
)

dsFile = nc4.Dataset("SouthChina_LandMask_4DL.nc")
ilon = np.copy(dsFile.variables["lon"])
ilat = np.copy(dsFile.variables["lat"])

lon_nc[:] = ilon
lat_nc[:] = ilat
time_nc[:] = itime
precip_nc[:, :, :] = precip

ncfile.close()
