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
# 13. test code to evaluate the function of reduce_mean() for the axis = none
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
M = 1


############################################################################
# train parameters
############################################################################
epochs = 6
epochs_switch_snr = 2
batch_size = 2

N_train = 100
N_test  = 10000
############################################################################
# Generate one hot encoded vector
############################################################################
# #generating data1 of size M
data1 = np.random.randint(M+1, size=N_train)
# data1 = tf.one_hot(
#     label1,
#     M,
#     on_value=1.0,
#     off_value=0.0,
#     axis=None,
#     dtype=None,
#     name=None
# )
# data1 = tf.constant([[1],[1],[1],[1],[1],[1],[1],[1],[1],[0] ],
#                 dtype=tf.float32)
print (data1)
print (data1.shape)

# #### defining autoencoder and it's layer
# ###########################################################
# TX part
# ###########################################################
input_signal = Input(shape=(M,))

# color constraints
color_loss = Lambda(lambda x: tf.math.reduce_mean(x, axis=0), name='color_loss')(input_signal)
# ###########################################################
# AE config
# ###########################################################
autoencoder = Model(inputs=[input_signal], outputs=[color_loss])
adam = Adam()  # SGD converge much slower than Adam

autoencoder.compile(optimizer=adam,
                    loss={
                        'color_loss': lambda y_true, y_pred: y_pred},
                    experimental_run_tf_function=False
                    )
print(autoencoder.summary())
# ###########################################################
# training auto encoder
# ###########################################################
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, mode='min')
reduce_learn_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=1)
random_const = np.random.randn(N_train, 1)
# constant = tf.expand_dims(tf.convert_to_tensor(1),axis=0)
autoencoder.fit(data1, [random_const],
                epochs=epochs,
                batch_size=batch_size
                )

############################################################################
# ### Make encoder and decoder
###########################################################################
# making encoder from full autoencoder
encoder_s1 = Model(input_signal, color_loss)
encoder_s1.predict([1])

