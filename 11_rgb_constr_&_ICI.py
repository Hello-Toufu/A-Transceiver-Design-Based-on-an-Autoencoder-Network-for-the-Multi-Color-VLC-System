############################################################################
# Author            :   ZDF
# Created on        :   11/26/2019 Tue
# last modified     :   11/29/2019 Tue
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
#     modified rgb eclipse constraints && sigmoid before transmit
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
M = 8
# n_channel = 4  # RGBY
n_channel = 3  # RGBY
k = np.log2(M)
k = int(k)

R = k / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)

############################################################################
# train parameters
############################################################################
epochs = 16
epochs_switch_snr = 2
batch_size = 5120

N_train = 10000000
N_test  = 10000
N_MonteCarlo = 1

# vlc parameter
A = 1  # peak amplitude
SNR_train_dB_1 = 15
SNR_train_dB_2 = 10

# dynamic alpha weights for two losses
alpha = K.variable(0.5) # initialization

############################################################################
# channel parameters
############################################################################
# 2-D array: 2 x 2
h11 = 1
h12 = 1
h21 = 1
h22 = 1
# H = np.array([h11 h12], [h21 h22])
xi = 0.1
print("xi:", xi)
H_rgby = tf.constant([[1 - xi, xi, 0, 0], [xi, 1 - 2 * xi, xi, 0], [0, xi, 1 - 2 * xi, xi], [0, 0, xi, 1 - xi]],
                dtype=tf.float32)
H_rgb = tf.constant([[1 - xi, xi, 0], [xi, 1 - 2 * xi, xi], [0, xi, 1 - xi]],
                dtype=tf.float32)
# H = H_rgby
H_propotion = tf.constant([[4.2730/1.1408, 0, 0], [0, 10.8603/1.3226, 0], [0, 0, 1]],
                dtype=tf.float32)

H = tf.matmul(H_propotion,H_rgb)
print("H=",H)

############################################################################
# constraints parameters
############################################################################
# parameter for Macadam eclipse
g11=86e4/1e2
g12=-40e4/1e2
g22=45e4/1e2
epsilon = 7/tf.sqrt(1e2)
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

dd = tf.keras.backend.mean(data1, axis=1)
cc = tf.abs(tf.keras.backend.mean(data1, axis=0))
data1
# print (data1)
print (data1.shape)

# #### defining autoencoder and it's layer
# ###########################################################
# TX part
# ###########################################################
input_signal = Input(shape=(M,))
# encoded_s_combine = Dense(M, activation='relu')(input_signal)
# print (data1[1,:])
encoded_s_combine = Dense(M, activation='linear')(input_signal)
# print (encoded_s_combine[1,:])
# encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
# encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
# encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
# print (encoded_s_combine[1,:])
# encoded_constraint = Dense(2 * n_channel, activation='relu')(encoded_s_combine)

# ###########################################################
# TX signal constraints
# ###########################################################
# print (encoded_s_combine[1,:])
# encoded_constraint = Dense(n_channel, activation='linear')(encoded_s_combine)
# softplus & hard_sigmoid & exponential (good)
# relu (bad) linear (wrong)
encoded_constraint = Dense(n_channel, activation='sigmoid')(encoded_s_combine)
# encoded_constraint = Dense(n_channel, activation='relu')(encoded_s_combine)
# encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,tf.math.maximum(encoded_s_combine,0.0)))(encoded_s_combine)
# encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,encoded_s_combine))(encoded_s_combine)
# print ("encoded_constraint",encoded_constraint[1,:])

# total optical power (lm) constraint added to loss
# total_o_power_train = 190
print ("encoded_constraint:",encoded_constraint.shape)
# mean of batch, sum of all led
# total_o_power_loss = Lambda(lambda x: tf.expand_dims(tf.abs(tf.reduce_sum(tf.keras.backend.mean(encoded_constraint, axis=0)) - total_o_power_train),axis=0), name='total_o_power_loss')\
#         (encoded_constraint)
# color constraints
I_mean = Lambda(lambda x: tf.math.reduce_mean(x, axis=0))\
            (encoded_constraint)
# I_mean_proportional = Lambda(lambda x: tf.math.reduce_mean(encoded_constraint, axis=0)/tf.reduce_sum(tf.math.reduce_mean(encoded_constraint, axis=0)))\
#             (encoded_constraint)

# optcial power propotion for different color
# L_propotional = I_mean/c/tf.reduce_sum(I_mean/c)

# total_o_power_loss = Lambda(lambda x: tf.expand_dims(tf.abs(tf.reduce_sum(I_mean/c) - total_o_power_train),axis=0), name='total_o_power_loss')\
#         (encoded_constraint)
# rgby constraints without eclipse
# x_mixed_rgby = tf.reduce_sum(xc_divide_yc*L_propotional)
# y_mixed_rgby = tf.reduce_sum(yc_reciprocal*L_propotional)
# color_loss = Lambda(lambda x: tf.expand_dims(
#                     tf.square(x_mixed_rgby-x_desired/y_desired) + tf.square(y_mixed_rgby - 1/y_desired) ,axis=0)
#                     , name='color_loss')(x_mixed_rgby,y_mixed_rgby)

x_mixed_rgby = tf.reduce_sum(xc_divide_yc*I_mean/c)/tf.reduce_sum(yc_reciprocal*I_mean/c)
y_mixed_rgby = tf.reduce_sum(I_mean/c)/tf.reduce_sum(yc_reciprocal*I_mean/c)
# color_loss = Lambda(lambda x: tf.expand_dims(tf.abs (
#                     g11*tf.square(x_mixed_rgby-x_desired) + 2 * g12 *(x_mixed_rgby-x_desired) * (y_mixed_rgby - y_desired) + g22 * tf.square(y_mixed_rgby - y_desired) - epsilon **2),axis=0)
#                     , name='color_loss')(x_mixed_rgby,y_mixed_rgby)
color_loss = Lambda(lambda x: tf.expand_dims(
                    g11*tf.square(x_mixed_rgby-x_desired) + 2 * g12 *(x_mixed_rgby-x_desired) * (y_mixed_rgby - y_desired) + g22 * tf.square(y_mixed_rgby - y_desired) ,axis=0)
                    , name='color_loss')(x_mixed_rgby,y_mixed_rgby)

# print ("total_o_power_loss:",total_o_power_loss.shape)
print ("color_loss:",color_loss.shape)
# peak intensity and nonnegativity intensity constraint
# encoded_constraint = Activation('sigmoid')(encoded_constraint)
# encoded_constraint = Lambda(lambda x: A * x)(encoded_constraint)

# ###########################################################
# Channel part
# ###########################################################
SNR_train = 10. ** (SNR_train_dB_1 / 10.)  # coverted 7 db of EbNo
# print ("SNR_train:",SNR_train )
# matrix multiply
rec_without_noise = tf.matmul(encoded_constraint,
                              H,
                              transpose_a=False,
                              transpose_b=True,
                              adjoint_a=False,
                              adjoint_b=False,
                              a_is_sparse=False,
                              b_is_sparse=False,
                              name=None)
# self-defined normalization
rec_without_noise_normalize = (rec_without_noise - tf.reduce_mean(rec_without_noise)) / tf.sqrt(
            tf.reduce_mean(
                (rec_without_noise - tf.reduce_mean(rec_without_noise)) ** 2
            )
        )

# print ("rec_without_noise_normalize:",rec_without_noise_normalize[1,:] )
rec_with_noise = GaussianNoise(
    np.sqrt(1 / SNR_train))(rec_without_noise_normalize)
# print ("rec_with_noise:",rec_with_noise[1,:] )
# np.sqrt(1 / SNR_train)

# ###########################################################
# RX part
# ###########################################################
decoded_s1 = Dense(M, activation='linear')(rec_with_noise)
# decoded_s1 = Dense(M, activation='linear')(decoded_s1)
# decoded_s1 = Dense(M, activation='linear')(decoded_s1)
# decoded_s1 = Dense(M, activation='relu')(decoded_s1)
# decoded_s1 = BatchNormalization(momentum=0, center=False, scale=False)(decoded_s1)
# decoded1_s1 = Dense(M, activation='relu', name='s1_softmax')(decoded_s1)
decoded1_s1 = Dense(M, activation='softmax', name='bler_loss')(decoded_s1)

# ###########################################################
# AE config
# ###########################################################
autoencoder = Model(inputs=[input_signal], outputs=[decoded1_s1, color_loss])
adam = Adam()  # SGD converge much slower than Adam
# adam = SGD()  # SGD converge much slower than Adam


# callback
class MyCallback(Callback):
    def __init__(self, alpha):
        self.alpha = alpha
        # self.beta  = beta
        # self.gamma = gamma

    # # customize your behavior
    # def on_batch_end(self, epoch, logs={}):
    #     results = [logs['bler_loss_loss'], logs['color_loss_loss']]
    #     # results = [1, 1]
    #     K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )

    def on_epoch_end(self, epoch, logs={}):
        results = [logs['bler_loss_loss'], logs['color_loss_loss']]
        K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
        print("\n epoch %s, alpha = %s" % (
            epoch + 1, K.get_value(self.alpha)))

autoencoder.compile(optimizer=adam,
                    loss={
                        'bler_loss': 'categorical_crossentropy',
                        'color_loss': lambda y_true, y_pred: y_pred},
                    loss_weights={
                        'bler_loss': alpha,
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
encoder_s1 = Model(input_signal, encoded_constraint)

# making decoder from full autoencoder
dec_input_1 = Input(shape=(n_channel,))
# deco_s1 = autoencoder.layers[-14](dec_input_1)
# deco_s1 = autoencoder.layers[-12](deco_s1)
# deco_s1 = autoencoder.layers[-10](deco_s1)
# deco_s1 = autoencoder.layers[-9](dec_input_1)
deco_s1 = autoencoder.layers[-5](dec_input_1)
deco_s1 = autoencoder.layers[-2](deco_s1)
# deco_s1 = autoencoder.layers[-1](deco_s1)
decoder_s1 = Model(dec_input_1, deco_s1)

############################################################################
# ### Visualize transmitted results
############################################################################
print("Visualize transmitted results")
scatter_point = []
for i in range(0, M):
    temp = np.zeros(M)
    temp[i] = 1
    # print(temp)
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

# minimum Euclidian distance
min_Euclidian_distance = 1000.0
for i in range(0, M):
    for j in range(i+1,M):
        Euclidian_distance_temp = tf.reduce_mean(tf.square( scatter_point[i,:,:] - scatter_point[j,:,:] ))
        if Euclidian_distance_temp < min_Euclidian_distance:
            min_Euclidian_distance = Euclidian_distance_temp
print ("min_Euclidian_distance:",min_Euclidian_distance)
# Visualize transmitted results use 3D figure
fig = plt.figure()
ax = Axes3D(fig)
xdata = scatter_point[:,0,0]
ydata = scatter_point[:,0,1]
zdata = scatter_point[:,0,2]
ax.scatter3D(xdata, ydata, zdata, s=100)
plt.xlabel('R',color='r')
plt.ylabel('G',color='g')
ax.set_zlabel('B',color='b')
plt.show()


############################################################################
# Autoencoder BLER(block error rate) performance
###########################################################################
SNR_range_dB = np.arange(0, 25, 5)
print("SNR_range_dB:", SNR_range_dB)

bler_s1_sum = np.zeros(len(SNR_range_dB))
for i_MonteCarlo in range(0, N_MonteCarlo):
    # ------------ generate one hot encoded vector for test-----------------
    test_label_s1 = np.random.randint(M, size=N_test)
    #     print("test_label_s1.shape",test_label_s1.shape)
    test_data1 = tf.one_hot(test_label_s1, M, on_value=1.0, off_value=0.0)
    #     print("test_data1.shape",test_data1.shape)
    nosie_normalize = np.random.randn(N_test, n_channel)
    #         print (encoder_s1.predict(test_data1).shape)
    # --------------------------------------------------------
    bler_s1 = [None] * len(SNR_range_dB)
    for n in range(0, len(SNR_range_dB)):
        SNR = 10.0 ** (SNR_range_dB[n] / 10.0)
        noise_std = np.sqrt(1 / SNR)
        noise_mean = 0
        noise = noise_std * nosie_normalize  # * 0
        test_r1_without_noise = tf.matmul(H,
                                          encoder_s1.predict(test_data1),
                                          transpose_a=False,
                                          transpose_b=True,
                                          adjoint_a=False,
                                          adjoint_b=False,
                                          a_is_sparse=False,
                                          b_is_sparse=False,
                                          name=None)
        test_r1_without_noise = tf.transpose(test_r1_without_noise)
        # normalize
        test_r1_without_noise = (test_r1_without_noise - tf.reduce_mean(test_r1_without_noise)) / tf.sqrt(
            tf.reduce_mean(
                (test_r1_without_noise - tf.reduce_mean(test_r1_without_noise)) ** 2
            )
        )
        #         print("test_r1_without_noise.shape",test_r1_without_noise.shape)
        pred_final_signal_s1 = decoder_s1.predict(test_r1_without_noise + noise)
        #         print("pred_final_signal_s1.shape",pred_final_signal_s1.shape)
        pred_output_s1 = tf.argmax(pred_final_signal_s1, axis=1)
        bler_s1[n] = sum(1. * (tf.cast(
            pred_output_s1 != tf.convert_to_tensor(test_label_s1, dtype=tf.int64), tf.float32
        ))) / N_test
    print("bler_s1:", bler_s1)
    bler_s1_sum = bler_s1_sum + bler_s1
bler_s1_mean = bler_s1_sum / N_MonteCarlo

print("s1 bler:", bler_s1_mean)

# #### plot BLER curve
plt.plot(SNR_range_dB, bler_s1_mean, 'bo-',
         label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')
# plt.plot(SNR_range_dB, bler_s2_mean, 'k*-',
#          label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s2')
plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid(True)
plt.legend(loc='upper right', ncol=1)
plt.xticks(SNR_range_dB, SNR_range_dB[::1])
plt.show()
print(mpl.get_backend())
# plt.xlim(SNR_range_dB.min() * 1, SNR_range_dB.max() * 1.1)
# plt.ylim(10 ** (-6), 1)

# print into a file
# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'a')
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger("D:\\12.txt", sys.stdout)
# sys.stderr = Logger(a.log_file, sys.stderr)		# redirect std err, if necessary
