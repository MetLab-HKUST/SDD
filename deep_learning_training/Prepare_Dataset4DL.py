# Preparing datasets for training the RxNet for rainfall statistics regression

# This script read reanalysis and rainfall observation data, shuffle them, and save
# in netCDF formats.


# Import libraries
import os
import netCDF4 as nc4
import numpy as np

# Load and prepare the dataset

# Load and shuffle rainfall data
dsFile = nc4.Dataset("DL_Precip_4NormalMax_Dist.nc")
precipAvg0 = dsFile.variables["precipAvg"]
precipAvg = np.copy(precipAvg0)
precipVar0 = dsFile.variables["precipVar"]
precipVar = np.copy(precipVar0)
precipMax0 = dsFile.variables["precipMax"]
precipMax = np.copy(precipMax0)
precip0 = np.stack((precipAvg, precipVar, precipMax), axis=-1)
# precip0.shape = (184072, 2)
idx = np.arange(precip0.shape[0])
np.random.shuffle(idx)    # shuffle the sample dimension
precip = np.copy(precip0[idx, :])
del precipAvg0, precipAvg, precipVar0, precipVar, precipMax0, precip0

# Load log precipitation rate data; eventually NOT used for the paper 
dsFile = nc4.Dataset("DL_Precip_4logNormal_Dist.nc")
logPrecipAvg0 = dsFile.variables["logPrecipAvg"]
logPrecipAvg = np.copy(logPrecipAvg0)
logPrecipVar0 = dsFile.variables["logPrecipVar"]
logPrecipVar = np.copy(logPrecipVar0)
logPrecip0 = np.stack((logPrecipAvg, logPrecipVar, precipMax), axis=-1)
# logPrecip0.shape = (184072, 2)
logPrecip = np.copy(logPrecip0[idx, :])
del logPrecipAvg0, logPrecipAvg, logPrecipVar0, logPrecipVar, logPrecip0, precipMax 

# Load circulation data, one by one
dsFile = nc4.Dataset("DL_ERA5_Geopotential.nc")
Z0 = dsFile.variables["Z"]
Z = np.copy(Z0)
Zsh = np.copy(Z[idx, :, :, :])
del Z, Z0
Zpe = np.transpose(Zsh, (0, 2, 3, 1))
# (sample, lat, lon, level)
del Zsh
LS = np.copy(Zpe)
del Zpe

dsFile = nc4.Dataset("DL_ERA5_SpecificHumidity.nc")
Q0 = dsFile.variables["Q"]
Q = np.copy(Q0)
Qsh = np.copy(Q[idx, :, :, :])
del Q, Q0
Qpe = np.transpose(Qsh, (0, 2, 3, 1))
# (sample, lat, lon, level)
del Qsh
LS = np.concatenate((LS, Qpe), axis=3)
del Qpe

dsFile = nc4.Dataset("DL_ERA5_SurfaceGeopotential.nc")
Zs0 = dsFile.variables["Zs"]
Zs = np.copy(Zs0)
ZsSh = np.copy(Zs[idx, :, :, :])
del Zs, Zs0
ZsPe = np.transpose(ZsSh, (0, 2, 3, 1))
# (sample, lat, lon, level)
del ZsSh
LS = np.concatenate((LS, ZsPe), axis=3)
del ZsPe

dsFile = nc4.Dataset("DL_ERA5_Temperature.nc")
T0 = dsFile.variables["T"]
T = np.copy(T0)
Tsh = np.copy(T[idx, :, :, :])
del T, T0
Tpe = np.transpose(Tsh, (0, 2, 3, 1))
# (sample, lat, lon, level)
del Tsh
LS = np.concatenate((LS, Tpe), axis=3)
del Tpe

dsFile = nc4.Dataset("DL_ERA5_Uwind.nc")
U0 = dsFile.variables["U"]
U = np.copy(U0)
Ush = np.copy(U[idx, :, :, :])
del U, U0
Upe = np.transpose(Ush, (0, 2, 3, 1))
# (sample, lat, lon, level)
del Ush
LS = np.concatenate((LS, Upe), axis=3)
del Upe

dsFile = nc4.Dataset("DL_ERA5_Vwind.nc")
V0 = dsFile.variables["V"]
V = np.copy(V0)
Vsh = np.copy(V[idx, :, :, :])
del V, V0
Vpe = np.transpose(Vsh, (0, 2, 3, 1))
# (sample, lat, lon, level)
del Vsh
LS = np.concatenate((LS, Vpe), axis=3)
del Vpe

# scale the large-scale (LS) patterns and save the scaling parameters
LSmean = np.mean(LS, axis=0, keepdims=True)
LSstd = np.std(LS, axis=(0, 1, 2), keepdims=True)
LSscaled = (LS - LSmean) / LSstd
del LS

filename = "DL_ERA5_Scaling_Parameters.nc"
if os.path.exists(filename):
    os.remove(filename)

ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 48)
ncfile.createDimension("lat", 48)
ncfile.createDimension("level", LSscaled.shape[3])

time_nc = ncfile.createVariable("time", np.float32, ("time",))
time_nc.long_name = "time"
lat_nc = ncfile.createVariable("lat", np.float32, ("lat",))
lat_nc.long_name = "lat"
lon_nc = ncfile.createVariable("lon", np.float32, ("lon",))
lon_nc.long_name = "lon"
lev_nc = ncfile.createVariable("level", np.float32, ("level",))
lat_nc.long_name = "level"

LSmean_nc = ncfile.createVariable(
    "LSmean", np.float32, ("time", "lat", "lon", "level"), fill_value=1.0e36
)
LSmean_nc.long_name = "time mean fields"

LSstd_nc = ncfile.createVariable(
    "LSstd", np.float32, ("time", "lat", "lon", "level"), fill_value=1.0e36
)
LSstd_nc.standard_name = "temporal-spatial standard deviation"

lon0 = np.copy(dsFile.variables["lon"])
lat0 = np.copy(dsFile.variables["lat"])
lev0 = np.arange(LSscaled.shape[3])

lon_nc[:] = lon0
lat_nc[:] = lat0
lev_nc[:] = lev0
time_nc[:] = np.array([0])

LSmean_nc[:, :, :, :] = LSmean[:, :, :, :]
LSstd_nc[:, :, :, :] = LSstd * np.ones(LSmean.shape)
ncfile.close()

# partition dataset into training, validation, testing, and save testing dataset as netCDF
# file for future use; save training and validation as TF dataset for next-step training
trainingEndId = int(precip.shape[0] * 0.7)
validationEndId = int(precip.shape[0] * 0.85)

trainingLS = LSscaled[0:trainingEndId, :, :, :]
validationLS = LSscaled[trainingEndId:validationEndId, :, :, :]
testingLS = LSscaled[validationEndId:, :, :, :]
del LSscaled

trainingPrecip = precip[0:trainingEndId, :]
validationPrecip = precip[trainingEndId:validationEndId, :]
testingPrecip = precip[validationEndId:, :]
del precip

trainingLogPrecip = logPrecip[0:trainingEndId, :]
validationLogPrecip = logPrecip[trainingEndId:validationEndId, :]
testingLogPrecip = logPrecip[validationEndId:, :]
del logPrecip

testingIdx = idx[validationEndId:]

# save testing dataset to a netCDF file, including the index
filename = "DL_ERA5_Testing_Scaled.nc"
if os.path.exists(filename):
    os.remove(filename)

ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 48)
ncfile.createDimension("lat", 48)
ncfile.createDimension("level", testingLS.shape[3])
ncfile.createDimension("moment", 3)

time_nc = ncfile.createVariable("time", np.float32, ("time",))
time_nc.long_name = "time"
lat_nc = ncfile.createVariable("lat", np.float32, ("lat",))
lat_nc.long_name = "lat"
lon_nc = ncfile.createVariable("lon", np.float32, ("lon",))
lon_nc.long_name = "lon"
lev_nc = ncfile.createVariable("level", np.float32, ("level",))
lat_nc.long_name = "level"
mom_nc = ncfile.createVariable("moment", np.float32, ("moment",))
lat_nc.long_name = "moment"

testingLS_nc = ncfile.createVariable(
    "testingLS", np.float32, ("time", "lat", "lon", "level"), fill_value=1.0e36
)
testingLS_nc.long_name = "Scaled large-scale fields for testing"

testingPrecip_nc = ncfile.createVariable(
    "testingPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
testingPrecip_nc.standard_name = "mean, variance, max of precipitation"

testingLogPrecip_nc = ncfile.createVariable(
    "testingLogPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
testingLogPrecip_nc.standard_name = "mean and variance of log precipitation and max precipitation"

testingIndex_nc = ncfile.createVariable(
    "testingIndex", np.float32, ("time",), fill_value=1.0e36
)
testingIndex_nc.standard_name = "index of samples in the original netCDF DL dataset"

lon0 = np.copy(dsFile.variables["lon"])
lat0 = np.copy(dsFile.variables["lat"])
lev0 = np.arange(testingLS.shape[3])
mom0 = np.array([1, 2, 3])

lon_nc[:] = lon0
lat_nc[:] = lat0
lev_nc[:] = lev0
mom_nc[:] = mom0
time_nc[:] = np.arange(testingLS.shape[0])

testingLS_nc[:, :, :, :] = testingLS
testingPrecip_nc[:, :] = testingPrecip
testingLogPrecip_nc[:, :] = testingLogPrecip
testingIndex_nc[:] = testingIdx

ncfile.close()

del testingLS

# save validation dataset to a netCDF file

filename = "DL_ERA5_Validation_Scaled.nc"
if os.path.exists(filename):
    os.remove(filename)

ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 48)
ncfile.createDimension("lat", 48)
ncfile.createDimension("level", validationLS.shape[3])
ncfile.createDimension("moment", 3)

time_nc = ncfile.createVariable("time", np.float32, ("time",))
time_nc.long_name = "time"
lat_nc = ncfile.createVariable("lat", np.float32, ("lat",))
lat_nc.long_name = "lat"
lon_nc = ncfile.createVariable("lon", np.float32, ("lon",))
lon_nc.long_name = "lon"
lev_nc = ncfile.createVariable("level", np.float32, ("level",))
lat_nc.long_name = "level"
mom_nc = ncfile.createVariable("moment", np.float32, ("moment",))
lat_nc.long_name = "moment"

validationLS_nc = ncfile.createVariable(
    "validationLS", np.float32, ("time", "lat", "lon", "level"), fill_value=1.0e36
)
validationLS_nc.long_name = "Scaled large-scale fields for validation"

validationPrecip_nc = ncfile.createVariable(
    "validationPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
validationPrecip_nc.standard_name = "mean, variance and max of precipitation"

validationLogPrecip_nc = ncfile.createVariable(
    "validationLogPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
validationLogPrecip_nc.standard_name = "mean and variance of log precipitation, and max precipitation"

lon0 = np.copy(dsFile.variables["lon"])
lat0 = np.copy(dsFile.variables["lat"])
lev0 = np.arange(validationLS.shape[3])
mom0 = np.array([1, 2, 3])

lon_nc[:] = lon0
lat_nc[:] = lat0
lev_nc[:] = lev0
mom_nc[:] = mom0
time_nc[:] = np.arange(validationLS.shape[0])

validationLS_nc[:, :, :, :] = validationLS
validationPrecip_nc[:, :] = validationPrecip
validationLogPrecip_nc[:, :] = validationLogPrecip

ncfile.close()

del validationLS

# save training dataset to a netCDF file

filename = "DL_ERA5_Training_Scaled.nc"
if os.path.exists(filename):
    os.remove(filename)

ncfile = nc4.Dataset(filename, mode="w", format="NETCDF4")
ncfile.createDimension("time", None)
ncfile.createDimension("lon", 48)
ncfile.createDimension("lat", 48)
ncfile.createDimension("level", trainingLS.shape[3])
ncfile.createDimension("moment", 3)

time_nc = ncfile.createVariable("time", np.float32, ("time",))
time_nc.long_name = "time"
lat_nc = ncfile.createVariable("lat", np.float32, ("lat",))
lat_nc.long_name = "lat"
lon_nc = ncfile.createVariable("lon", np.float32, ("lon",))
lon_nc.long_name = "lon"
lev_nc = ncfile.createVariable("level", np.float32, ("level",))
lat_nc.long_name = "level"
mom_nc = ncfile.createVariable("moment", np.float32, ("moment",))
lat_nc.long_name = "moment"

trainingLS_nc = ncfile.createVariable(
    "trainingLS", np.float32, ("time", "lat", "lon", "level"), fill_value=1.0e36
)
trainingLS_nc.long_name = "Scaled large-scale fields for training"

trainingPrecip_nc = ncfile.createVariable(
    "trainingPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
trainingPrecip_nc.standard_name = "mean, variance, and max of precipitation"

trainingLogPrecip_nc = ncfile.createVariable(
    "trainingLogPrecip", np.float32, ("time", "moment"), fill_value=1.0e36
)
trainingLogPrecip_nc.standard_name = "mean and variance of log precipitation and max precipitation"

lon0 = np.copy(dsFile.variables["lon"])
lat0 = np.copy(dsFile.variables["lat"])
lev0 = np.arange(trainingLS.shape[3])
mom0 = np.array([1, 2, 3])

lon_nc[:] = lon0
lat_nc[:] = lat0
lev_nc[:] = lev0
mom_nc[:] = mom0
time_nc[:] = np.arange(trainingLS.shape[0])

trainingLS_nc[:, :, :, :] = trainingLS
trainingPrecip_nc[:, :] = trainingPrecip
trainingLogPrecip_nc[:, :] = trainingLogPrecip

ncfile.close()
