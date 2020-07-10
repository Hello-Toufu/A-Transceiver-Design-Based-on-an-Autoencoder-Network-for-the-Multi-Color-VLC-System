############################################################################
# Author            :   ZDF
# Created on        :   2019
# last modified     :
# Description       :
# 1. basic frame for multi-color VLC system, ICI is set to be nearly zero
# 2. set ICI as 0.1/0.3
# 3. rgb eclipse constraints is add into consideration
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

import sys
# import keras
import logging
import os
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

############################################################################
# GPU config
############################################################################
# from tensorflow.python.client import device_lib
# import tensorflow.keras.backend.tensorflow_backend as KTF #进行配置，每个GPU使用80%上限现存

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # prohibit GPU set value = -1
# config = tf.compat.v1.ConfigProto() #tf 2.0 has no tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# # config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# tf.__version__
# # KTF.set_session(session )

# 列出所有的本地机器设备
# local_device_protos = device_lib.list_local_devices()
# 打印
#     print(local_device_protos)

# 只打印GPU设备
# [print(x) for x in local_device_protos if x.device_type == 'GPU']

# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)
# tf.test.is_gpu_available()

# from tensorflow.python.client import device_lib
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"

# if __name__ == "__main__":
#     print(device_lib.list_local_devices())

############################################################################
# system parameters
############################################################################
# * define (n_channel,k) here for (n,k) autoencoder
M = 16
n_channel = 3  # RGBY
k = np.log2(M)
k = int(k)

R = k / n_channel
print('M:', M, 'k:', k, 'n:', n_channel)

############################################################################
# train parameters
############################################################################
epochs = 10
epochs_switch_snr = 2
batch_size = 64

N_train = 100000
N_test = 10000
N_MonteCarlo = 1

# vlc parameter
A = 1  # peak amplitude
SNR_train_dB_1 = 10
SNR_train_dB_2 = 10

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
H = H_rgb
print("H=",H)

############################################################################
# Generate one hot encoded vector
############################################################################
# #generating data1 of size M
label1 = np.random.randint(M, size=N_train)
# plt.plot(range(0,N_train), label1, 'bo-',
#          label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')
# plt.plot(SNR_range_dB, bler_s2_mean, 'k*-',
#          label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s2')
# plt.grid(True)
# plt.show()
# print(mpl.get_backend())
# tf.convert_to_tensor(label1)
# label1.shape
# print (label1)
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
# data1.numpy()
# print (data1**2)

# tf.convert_to_tensor(label1,dtype=tf.int64)
# pred_output_s1 = tf.argmax(data1, axis=1)
# pred_output_s1
# pred_output_s1 != tf.convert_to_tensor(label1,dtype=tf.int64)
# tf.bitcast(pred_output_s1 != tf.convert_to_tensor(label1,dtype=tf.int64),tf.int64)
# tf.cast(pred_output_s1 != tf.convert_to_tensor(label1,dtype=tf.int64),tf.float32)
# (pred_output_s1 != tf.convert_to_tensor(label1,dtype=tf.int64)).numpy+0
# sum(1. * (tf.cast(pred_output_s1 != tf.convert_to_tensor(label1,dtype=tf.int64),tf.float32))) / N_test

# #### defining autoencoder and it's layer
# ###########################################################
# TX part
# ###########################################################
input_signal = Input(shape=(M,))
# encoded_s_combine = Dense(M, activation='relu')(input_signal)
# print (data1[1,:])
encoded_s_combine = Dense(M, activation='linear')(input_signal)
# print (encoded_s_combine[1,:])
encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
encoded_s_combine = Dense(M , activation='linear')(encoded_s_combine)
encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
# print (encoded_s_combine[1,:])
# encoded_constraint = Dense(2 * n_channel, activation='relu')(encoded_s_combine)

# ###########################################################
# TX signal constraints
# ###########################################################
# print (encoded_s_combine[1,:])
encoded_constraint = Dense(n_channel, activation='sigmoid')(encoded_s_combine)
# print ("encoded_constraint",encoded_constraint[1,:])

# average power constraint added to loss
aveP = 2
aveP_2 = 1
print ("encoded_constraint:",encoded_constraint.shape)
# mean of batch, sum of all led
# aveP_loss = Lambda(lambda x: tf.square(tf.reduce_sum(tf.keras.backend.mean(encoded_constraint, axis=0)) - aveP), name='aveP_loss')\
#             (encoded_constraint)
aveP_loss = Lambda(lambda x: tf.expand_dims(tf.abs(tf.reduce_sum(tf.keras.backend.mean(encoded_constraint, axis=0)) - aveP),axis=0), name='aveP_loss')\
        (encoded_constraint)
# color constraints
I_mean = Lambda(lambda x: tf.math.reduce_mean(encoded_constraint, axis=0))\
            (encoded_constraint)
x_old = np.array([0.7006,0.1547,0.1440])
y_old = np.array([0.2993,0.8059,0.0497])
xc_divide_yc  = tf.convert_to_tensor(x_old/y_old,dtype=tf.float32)
yc_reciprocal = tf.convert_to_tensor(np.array([1,1,1])/y_old,dtype=tf.float32)
g11=86e4
g12=-40e4
g22=45e4
epsilon = 7
c = tf.convert_to_tensor(np.array([0.0114,0.0052,0.0427]),dtype=tf.float32)

x_mixed = tf.math.divide(tf.reduce_sum(xc_divide_yc*I_mean/c,axis=0),tf.reduce_sum(yc_reciprocal*I_mean/c,axis=0))
y_mixed = tf.math.divide(tf.reduce_sum(I_mean/c,axis=0),tf.reduce_sum(yc_reciprocal*I_mean/c,axis=0))
color_loss = Lambda(lambda x: tf.expand_dims(tf.abs (
                    g11*tf.square(x_mixed-0.313) + 2 * g12 *(x_mixed-0.313) * (y_mixed - 0.337) + g22 * tf.square(y_mixed - 0.337) - epsilon **2),axis=0)
                    , name='color_loss')(x_mixed,y_mixed)
# x_mixed =
# print ("tf.keras.backend.mean(encoded_constraint, axis=0):",tf.keras.backend.mean(encoded_constraint, axis=0).shape)
print ("aveP_loss:",aveP_loss.shape)
print ("color_loss:",color_loss.shape)
# print (tf.convert_to_tensor(aveP).shape)
# print (tf.keras.backend.mean(encoded_s_combine,axis=1))
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
rec_without_noise_normalize = BatchNormalization(momentum=0.9, center=False, scale=False)(rec_without_noise)
# 借鉴优化算法里的momentum算法将历史batch里的mean和variance的作用延续到当前batch. 一般momentum的值为0.9 , 0.99等.
# 多个batch后, 即多个0.9连乘后,最早的batch的影响会变弱
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
autoencoder = Model(inputs=[input_signal], outputs=[decoded1_s1, aveP_loss, color_loss])
adam = Adam()  # SGD converge much slower than Adam
# adam = SGD()  # SGD converge much slower than Adam
# ###########################################################
# callback
# ###########################################################
# dynamic alpha weights for two losses
# SNR_train_dB = K.variable(SNR_train_dB_1)
# dynamic alpha weights for average power loss and BLER loss
# alpha = K.variable(0.5)
alpha = K.variable(0.3)
beta = K.variable(0.3)
gamma = K.variable(0.4)


class MyCallback(Callback):
    def __init__(self, alpha,beta):
        self.alpha = alpha
        self.beta  = beta
        # self.gamma = gamma

    # customize your behavior
    def on_batch_end(self, epoch, logs={}):
        results = [logs['bler_loss_loss'], logs['aveP_loss_loss'], logs['color_loss_loss']]
        # results = [1, 1]
        K.set_value(self.alpha,  results[0] / (results[0] + results[1] + results[2]) )
        K.set_value(self.beta,   results[1] / (results[0] + results[1] + results[2]) )
        # K.set_value(self.gamma,  1-self.alpha-self.beta )
        # K.set_value(self.alpha,  1.0 )
        # K.set_value(self.alpha, 0.5)
        # print (tf.keras.backend.mean(logs['encoded_constraint'], axis=0))

    def on_epoch_end(self, epoch, logs={}):
        print("\n epoch %s, alpha = %s, beta = %s, gamma = %s" % (
            epoch + 1, K.get_value(self.alpha), K.get_value(self.beta), 1-K.get_value(self.alpha)-K.get_value(self.beta)))

# autoencoder.compile(optimizer=adam,
#                    loss='categorical_crossentropy'
#                    )
autoencoder.compile(optimizer=adam,
                    loss={
                        'bler_loss': 'categorical_crossentropy',
                        'aveP_loss': lambda y_true, y_pred: y_pred,
                        'color_loss': lambda y_true, y_pred: y_pred},
                    loss_weights={
                        'bler_loss': alpha,
                        'aveP_loss': beta,
                        'color_loss': 1 - alpha - beta},
                    experimental_run_tf_function=False
                    )
# printing summary of layers and it's trainable parameters
print(autoencoder.summary())
# ###########################################################
# training auto encoder
# ###########################################################
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, mode='min')
reduce_learn_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=1)
random_aveP = np.random.randn(N_train, 1)
# constant = tf.expand_dims(tf.convert_to_tensor(1),axis=0)
autoencoder.fit(data1, [data1, random_aveP, random_aveP],
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[reduce_learn_rate
                            ,early_stopping
                            ,MyCallback(alpha,beta)
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
deco_s1 = autoencoder.layers[-3](dec_input_1)
deco_s1 = autoencoder.layers[-2](deco_s1)
# deco_s1 = autoencoder.layers[-1](deco_s1)
decoder_s1 = Model(dec_input_1, deco_s1)

# ### Visualize transmitted results
scatter_point = []
for i in range(0, M):
    temp = np.zeros(M)
    temp[i] = 1
    scatter_point.append(encoder_s1.predict(np.expand_dims(temp, axis=0)))
scatter_point = np.array(scatter_point)
print(scatter_point.shape)
print(scatter_point)
print ("average power:",np.mean(scatter_point, axis=0)  )
print ("average power loss:",np.sum(np.mean(scatter_point, axis=0)) - aveP)

# ### Autoencoder BLER(block error rate) performance
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
