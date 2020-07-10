############################################################################
# Author            :   ZDF
# Created on        :   2019
# last modified     :   11/26/2019 Tue
# Description       :
# 1. basic frame for multi-color VLC system, ICI is set to be nearly zero
# 2. set ICI as 0.1/0.3
# 3. rgb eclipse constraints is add into consideration
# 4. rgby eclipse constraints (bad)
# 5. rgby constraints without Macadam eclipse (good, big batch size to achieve more stable performance))
# 7. total optical power constr replace the total current constr / the normalization before adding noise at receiver  (good)
# 8. remove the total optical power constraints (it doesn‘t represent SNR)  (good)
# 9. rgby eclipse constraints without the total optical power constraints (good)
# 10. rgb eclipse constraints without the total optical power constraints (bad)
# 11. rgb fixed propotion constraints && channel matrix with RGB efficiency (good) && replace sigmoid with linear
# 14. test code to evaluate the function of reduce_mean() for the axis = none & loss caculation
############################################################################

############################################################################
# ### Import libs
############################################################################
# * using tensorflow2.0(us tf.keras rather than pure keras)
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d.axes3d import Axes3D

import sys
# import keras
import logging
import os
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

############################################################################
# system parameters
############################################################################
# * define (n_channel,k) here for (n,k) autoencoder
M = 3
# n_channel = 4  # RGBY
n_channel = 3  # RGBY
k = np.log2(M)
k = int(k)

R = k / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)

############################################################################
# train parameters
############################################################################
epochs = 6
epochs_switch_snr = 2
batch_size = 5120

N_train = 1000000
N_test  = 10000
N_MonteCarlo = 1
# dynamic alpha weights for two losses
alpha = K.variable(0.9) # initialization




############################################################################
# constraints parameters
############################################################################
# parameter for Macadam eclipse
g11=86e4/1e1
g12=-40e4/1e1
g22=45e4/1e1
epsilon = 7/tf.sqrt(1e1)
# parameter for desired CCT value
x_desired = 0.313
y_desired = 0.337
x_old = np.array([0.7006,0.1547,0.1440])
y_old = np.array([0.2993,0.8059,0.0497])
xc_divide_yc  = tf.convert_to_tensor(x_old/y_old,dtype=tf.float32)
yc_reciprocal = tf.convert_to_tensor(np.array([1,1,1])/y_old,dtype=tf.float32)
# c_rgby = tf.convert_to_tensor(np.array([0.021,0.014,0.005,0.015]),dtype=tf.float32)
c_rgb = tf.convert_to_tensor(np.array([0.0114,0.0052,0.0427]),dtype=tf.float32)
c = c_rgb

############################################################################
# Generate one hot encoded vector
############################################################################
# #generating data1 of size M
label1 = np.random.randint(M, size=N_train)
data1 = tf.one_hot(
    label1,
    M,
    on_value=1.0,
    off_value=0.0,
    axis=None,
    dtype=None,
    name=None
)
# scatter_point = []
# for i in range(0, M):
#     temp = np.zeros(M)
#     temp[i] = 1
#     print(temp)
#     scatter_point.append(np.expand_dims(temp, axis=0))
# data1 = np.array(scatter_point)
# data1 = data1[:,0,:]
# print (data1)
print (data1.shape)

# #### defining autoencoder and it's layer
# ###########################################################
# TX part
# ###########################################################
input_signal = Input(shape=(M,))
# encoded_s_combine = Dense(M, activation='linear')(input_signal)
# encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)


# encoded_constraint = Dense(n_channel, activation='sigmoid')(encoded_s_combine)
decoded1_s1 = Lambda(lambda x: x, name='bler_loss')\
            (input_signal)

# total_o_power_train = 190
# print ("encoded_constraint:",encoded_constraint.shape)
I_mean = Lambda(lambda x: tf.math.reduce_mean(x, axis=0))\
            (input_signal)

x_mixed_rgby = tf.reduce_sum(xc_divide_yc*I_mean/c)/tf.reduce_sum(yc_reciprocal*I_mean/c)
y_mixed_rgby = tf.reduce_sum(I_mean/c)/tf.reduce_sum(yc_reciprocal*I_mean/c)
color_loss = Lambda(lambda x: tf.expand_dims(
                    g11*tf.square(x_mixed_rgby-x_desired) + 2 * g12 *(x_mixed_rgby-x_desired) * (y_mixed_rgby - y_desired) + g22 * tf.square(y_mixed_rgby - y_desired) ,axis=0)
                    , name='color_loss')(x_mixed_rgby,y_mixed_rgby)

# print ("total_o_power_loss:",total_o_power_loss.shape)
print ("color_loss:",color_loss.shape)

# ###########################################################
# AE config
# ###########################################################
autoencoder = Model(inputs=[input_signal], outputs=[decoded1_s1, color_loss])
adam = Adam()  # SGD converge much slower than Adam


# callback
class MyCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
    def on_epoch_end(self, epoch, logs={}):
        results = [logs['bler_loss_loss'], logs['color_loss_loss']]
        K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
        print("\n epoch %s, alpha = %s" % (
            epoch + 1, K.get_value(self.alpha)))


# autoencoder.compile(optimizer=adam,
#                    loss='categorical_crossentropy'
#                    )
autoencoder.compile(optimizer=adam,
                    loss={
                        'bler_loss': 'categorical_crossentropy',
                        # 'total_o_power_loss': lambda y_true, y_pred: y_pred,
                        'color_loss': lambda y_true, y_pred: y_pred},
                    loss_weights={
                        'bler_loss': alpha,
                        # 'total_o_power_loss': beta,
                        'color_loss': 1 - alpha},
                    experimental_run_tf_function=False
                    )
# printing summary of layers and it's trainable parameters
print(autoencoder.summary())
# ###########################################################
# training auto encoder
# ###########################################################
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, mode='min')
reduce_learn_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=1)
random_const = np.random.randn(N_train, 1)
# constant = tf.expand_dims(tf.convert_to_tensor(1),axis=0)
autoencoder.fit(data1, [data1, random_const],
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[reduce_learn_rate
                            ,early_stopping
                            ,MyCallback(alpha)
                            ]
                )

############################################################################
# ### Make encoder and decoder
###########################################################################
# making encoder from full autoencoder
encoder_s1 = Model(input_signal, decoded1_s1)
# encoder_s2 = Model(input_signal, x_mixed_rgby)
# encoder_s3 = Model(input_signal, y_mixed_rgby)

############################################################################
# ### Visualize transmitted results
############################################################################
print("Visualize transmitted results")
scatter_point = []
for i in range(0, M):
    temp = np.zeros(M)
    temp[i] = 1
    print(temp)
    scatter_point.append(encoder_s1.predict(np.expand_dims(temp, axis=0)))
scatter_point = np.array(scatter_point)
print(scatter_point.shape)
print(scatter_point)
print ("average power:",np.mean(scatter_point, axis=0)  )

x_mixed_final = tf.reduce_sum(xc_divide_yc*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)/tf.reduce_sum(yc_reciprocal*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)
y_mixed_final = tf.reduce_sum(np.mean(scatter_point, axis=0)[0,:]/c,axis=0)/tf.reduce_sum(yc_reciprocal*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)
print("x_mixed_final:",x_mixed_final)
print("y_mixed_final:",y_mixed_final)
print("Macadam eclipse constraints:",g11 * tf.square(x_mixed_final - 0.313) + 2 * g12 * (x_mixed_final - 0.313) * (y_mixed_final - 0.337) + g22 * tf.square(
    y_mixed_final - 0.337) )

g11=86e4
g12=-40e4
g22=45e4
epsilon = 7
print("Macadam eclipse constraints (real):",g11 * tf.square(x_mixed_final - 0.313) + 2 * g12 * (x_mixed_final - 0.313) * (y_mixed_final - 0.337) + g22 * tf.square(
    y_mixed_final - 0.337) )

