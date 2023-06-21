# Training the RxNet for rainfall statistics regression

# This script read reanalysis and rainfall momentum (mean and variance) and train the
# RxNet for predicting the mean and variance of precipitation in a 1 degree by 1 degree
# grid cell, which contains 4x4 subgrid cells.


# Import libraries
import os
import time
import tensorflow as tf
from tensorflow import keras
# import tensorflow_probability as tfp
import netCDF4 as nc4
import numpy as np

# Load and prepare the dataset
dsFile = nc4.Dataset("DL_ERA5_Training_Scaled.nc")
trainLS0 = dsFile.variables["trainingLS"]
trainMOM0 = dsFile.variables["trainingPrecip"]
trainLS = np.copy(trainLS0)
trainMOM = np.copy(trainMOM0[:, 2]).reshape((trainLS.shape[0], 1))
idx = np.arange(trainLS.shape[0])

dsFile = nc4.Dataset("DL_ERA5_Validation_Scaled.nc")
validLS0 = dsFile.variables["validationLS"]
validMOM0 = dsFile.variables["validationPrecip"]
validLS = np.copy(validLS0)
validMOM = np.copy(validMOM0[:, 2]).reshape((validLS.shape[0], 1))


# Define the RxNet networks
def entry_flow(inputs):
    # Entry block
    x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(128, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for size in [256, 512, 728]:
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(size, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(size, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Project residual
        residual = keras.layers.Conv2D(
            size, 1, strides=2, padding='same')(previous_block_activation)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    return x


# # show model structure
# inputs = keras.Input(shape=(48, 48, 31))
# intermediate_outputs = entry_flow(inputs)
# intermediate_model = keras.Model(inputs, intermediate_outputs)
# intermediate_model.summary(line_length=120)


def middle_flow(x, num_blocks=8):
    previous_block_activation = x

    for _ in range(num_blocks):
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.add([x, previous_block_activation])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    return x


# # show the flow so far
# inputs = keras.Input(shape=(6, 6, 728))
# intermediate_outputs = middle_flow(inputs)
# intermediate_model = keras.Model(inputs, intermediate_outputs)
# intermediate_model.summary(line_length=120)


def exit_flow(x):
    previous_block_activation = x

    x = keras.layers.Activation('relu')(x)
    x = keras.layers.SeparableConv2D(728, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation('relu')(x)
    x = keras.layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Project residual
    residual = keras.layers.Conv2D(
        1024, 1, strides=2, padding='same')(previous_block_activation)
    x = keras.layers.add([x, residual])  # Add back residual

    x = keras.layers.SeparableConv2D(1536, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.SeparableConv2D(2048, 3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(1, activation=None)(x)

    return x


# inputs = keras.Input(shape=(6, 6, 728))
# outputs = exit_flow(inputs)
# intermediate_model = keras.Model(inputs, outputs)
# intermediate_model.summary(line_length=120)


# Create RxNet by chaining the 3 flows
Inputs = keras.Input(shape=(48, 48, 31))
Outputs = exit_flow(middle_flow(entry_flow(Inputs)))
RxNet = keras.Model(Inputs, Outputs)


# Define the loss function and the optimizer
def compute_loss(model, ls, obs):
    mom_pred = model(ls)  # actually max in this script
    # mean_pred, var_pred = tf.split(mom_pred, num_or_size_splits=2, axis=1)
    # var_pred = tf.math.exp(log_var_pred)
    mom_obs = obs
    # mean_obs, var_obs = tf.split(obs, num_or_size_splits=2, axis=1)
    # var_obs = var_obs + 1.0e-4   # add a small number to avoid dividing by zero in the DL loss
    # dl_loss = 0.5 * (
    #            var_pred / var_obs - 1 + (mean_obs - mean_pred) ** 2 / var_obs - tf.math.log(var_pred / var_obs))
    # mse_loss = (mean_pred - mean_obs)**2 + 0.5 * (var_pred - var_obs)**2
    mse_loss = (mom_pred - mom_obs) ** 2
    return tf.reduce_mean(mse_loss)


@tf.function
def train_step(model, ls, obs, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, ls, obs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# Train the model and save the best
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=6000,
    decay_rate=0.9)
zOptimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
epochs = 120
batch_size = 64
trainSubNum = 8
validSubNum = 3
trainSeg = int(trainLS.shape[0] / trainSubNum)
validSeg = int(validLS.shape[0] / validSubNum)
Best = 1.0e36
lossLog = np.zeros((epochs, 2))

for epoch in range(1, epochs + 1):
    start_time = time.time()
    trainLoss = tf.keras.metrics.Mean()
    for subset in range(trainSubNum):
        # Set up DL dataset
        trainLS_dataset = tf.data.Dataset.from_tensor_slices(
            trainLS[subset * trainSeg:(subset + 1) * trainSeg, :, :, :]).batch(batch_size)
        trainMOM_dataset = tf.data.Dataset.from_tensor_slices(
            trainMOM[subset * trainSeg:(subset + 1) * trainSeg, :]).batch(batch_size)
        for (LS, MOM) in zip(trainLS_dataset, trainMOM_dataset):
            train_loss = train_step(RxNet, LS, MOM, zOptimizer)
            trainLoss(train_loss)

    end_time = time.time()

    testLoss = tf.keras.metrics.Mean()
    for subset in range(validSubNum):
        validLS_dataset = tf.data.Dataset.from_tensor_slices(
            validLS[subset * validSeg:(subset + 1) * validSeg, :, :, :]).batch(batch_size)
        validMOM_dataset = tf.data.Dataset.from_tensor_slices(
            validMOM[subset * validSeg:(subset + 1) * validSeg, :]).batch(batch_size)
        for (LS, MOM) in zip(validLS_dataset, validMOM_dataset):
            test_loss = compute_loss(RxNet, LS, MOM)
            testLoss(test_loss)

    Current = testLoss.result()
    lossLog[epoch - 1, 0] = trainLoss.result()
    lossLog[epoch - 1, 1] = Current
    print("Epoch: {}".format(epoch))
    print("  Training Loss:       {:12.6f}".format(lossLog[epoch - 1, 0]))
    print("  Testing Loss:        {:12.6f}".format(Current))
    print("  Time elapse:         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf RxNet_Max_Only.h5")
        RxNet.save("RxNet_Max_Only.h5")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))

    np.random.shuffle(idx)
    trainLS = trainLS[idx, :, :, :]
    trainMOM = trainMOM[idx, :]


np.save('LossLog_Max_Only.npy', lossLog)
