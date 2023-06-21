# Training the RxNet for rainfall maximum distribution regression

# This script read reanalysis and rainfall momentum (mean and variance) and train the
# RxNet for predicting the maximum of precipitation in a 1 degree by 1 degree
# grid cell, which contains 4x4 subgrid cells.


# Import libraries
import os
import time
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import netCDF4 as nc4
import numpy as np


# Load and prepare the dataset
dsFile = nc4.Dataset("DL_ERA5_Training_Scaled.nc")
trainLS0 = dsFile.variables["trainingLS"]
trainMOM0 = dsFile.variables["trainingPrecip"]
trainLS = np.copy(trainLS0[0:128000, :, :, :])
trainMOM = np.copy(trainMOM0[0:128000, 2]).reshape((128000, 1))
idx = np.arange(trainLS.shape[0])

dsFile = nc4.Dataset("DL_ERA5_Validation_Scaled.nc")
validLS0 = dsFile.variables["validationLS"]
validMOM0 = dsFile.variables["validationPrecip"]
validLS = np.copy(validLS0[0:26624, :, :, :])
validMOM = np.copy(validMOM0[0:26624, 2]).reshape((26624, 1))

# Set the parallel training strategy
strategy = tf.distribute.MirroredStrategy()
batch_size = 8
GLOBAL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync


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
    x = keras.layers.Dense(2, activation=None)(x)

    prob_layer = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])))

    x = prob_layer(x)

    return x


# inputs = keras.Input(shape=(6, 6, 728))
# outputs = exit_flow(inputs)
# intermediate_model = keras.Model(inputs, outputs)
# intermediate_model.summary(line_length=120)


# Create RxNet by chaining the 3 flows
def create_model():
    inputs = keras.Input(shape=(48, 48, 31))
    outputs = exit_flow(middle_flow(entry_flow(inputs)))
    rxnet = keras.Model(inputs, outputs)

    return rxnet


with strategy.scope():
    def compute_loss(model, ls, obs):
        loss = tfp.experimental.nn.losses.negloglik(ls, obs, model, axis=-1)

        return tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)


with strategy.scope():
    zModel = create_model()
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=2000,
    #     decay_rate=0.9)
    zOptimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)


def train_step(model, ls, obs):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, ls, obs)

    gradients = tape.gradient(loss, model.trainable_variables)
    zOptimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test_step(model, ls, obs):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    loss = compute_loss(model, ls, obs)
    return loss


# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(model, ls, obs):
    per_replica_losses = strategy.run(train_step, args=(model, ls, obs))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_test_step(model, ls, obs):
    per_replica_losses = strategy.run(test_step, args=(model, ls, obs))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


# Train the model and save the best
epochs = 100
trainSubNum = 25
validSubNum = 13
trainSeg = int(trainLS.shape[0] / trainSubNum)
validSeg = int(validLS.shape[0] / validSubNum)
Best = 1.0e36
lossLog = np.zeros(epochs)

for epoch in range(1, epochs + 1):
    start_time = time.time()

    trainLoss = 0.0
    numBatches = 0
    for subset in range(trainSubNum):
        # Set up DL dataset
        trainLS_dataset = tf.data.Dataset.from_tensor_slices(
            trainLS[subset * trainSeg:(subset + 1) * trainSeg, :, :, :]).batch(GLOBAL_BATCH_SIZE)
        trainMOM_dataset = tf.data.Dataset.from_tensor_slices(
            trainMOM[subset * trainSeg:(subset + 1) * trainSeg, :]).batch(GLOBAL_BATCH_SIZE)

        trainLS_dist_dataset = strategy.experimental_distribute_dataset(trainLS_dataset)
        trainMOM_dist_dataset = strategy.experimental_distribute_dataset(trainMOM_dataset)

        for (LS, OBS) in zip(trainLS_dist_dataset, trainMOM_dist_dataset):
            trainLoss += distributed_train_step(zModel, LS, OBS)
            numBatches += 1
    
    trainLoss = trainLoss / numBatches

    end_time = time.time()

    validLoss = 0.0
    numBatches = 0
    for subset in range(validSubNum):
        validLS_dataset = tf.data.Dataset.from_tensor_slices(
            validLS[subset * validSeg:(subset + 1) * validSeg, :, :, :]).batch(GLOBAL_BATCH_SIZE)
        validMOM_dataset = tf.data.Dataset.from_tensor_slices(
            validMOM[subset * validSeg:(subset + 1) * validSeg, :]).batch(GLOBAL_BATCH_SIZE)

        validLS_dist_dataset = strategy.experimental_distribute_dataset(validLS_dataset)
        validMOM_dist_dataset = strategy.experimental_distribute_dataset(validMOM_dataset)

        for (LS, OBS) in zip(validLS_dist_dataset, validMOM_dist_dataset):
            validLoss += distributed_test_step(zModel, LS, OBS)
            numBatches += 1

    validLoss = validLoss / numBatches

    Current = validLoss
    lossLog[epoch - 1] = Current
    print("Epoch: {}".format(epoch))
    print("  Training Loss:         {:12.6f}".format(trainLoss))
    print("  Test Loss    :         {:12.6f}".format(Current))
    print("  Time elapse  :         {:12.6f}".format(end_time - start_time))

    if Current < Best:
        Best = Current
        os.system("rm -rf RxNet_Max_TFP_Normal.h5")
        zModel.save("RxNet_MaxPob_TFP_Normal.h5")
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        print("+++ New model saved")
    else:
        print("--- Best loss so far: {}".format(Best))

    np.random.shuffle(idx)
    trainLS = trainLS[idx, :, :, :]
    trainMOM = trainMOM[idx, :]


np.save('LossLog_Max_TFP_Normal.npy', lossLog)
