############################################################################
# Author            :   ZDF
# Created on        :   12/04/2019 Tue
# last modified     :   12/04/2019 Mon
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
# 15. muliple CCT value are considered
# 17. consider input-dependent noise (may be wrong, uncertainly)
# 18. input-dependent noise (not wrong, but performance worse than constellation design)
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
epochs = 10
epochs_switch_snr = 2
batch_size = 5120

N_train = 10000000
N_test  = 10000
N_MonteCarlo = 1

# vlc parameter
A = 1  # peak amplitude
# SNR_train_dB_1 = 15
# SNR_train_dB_2 = 10
SNR_train_dB_1 = np.random.uniform(5,15,size=N_train)
# dynamic alpha weights for two losses
alpha = K.variable(1.0) # initialization
beta = K.variable(0.0)

# signal-dependent noise
psi_2 = 5
############################################################################
# channel parameters
############################################################################
xi = 0.1
print("xi:", xi)
H_rgby = tf.constant([[1 - xi, xi, 0, 0], [xi, 1 - 2 * xi, xi, 0], [0, xi, 1 - 2 * xi, xi], [0, 0, xi, 1 - xi]],
                dtype=tf.float32)
H_rgb = tf.constant([[1 - xi, xi, 0], [xi, 1 - 2 * xi, xi], [0, xi, 1 - xi]],
                dtype=tf.float32)
# H = H_rgby
H_propotion = tf.constant([[3.7456, 0, 0], [0, 8.2115, 0], [0, 0, 1]],
                dtype=tf.float32)

H = tf.matmul(H_propotion,H_rgb)
H = H_rgb
print("H=",H)

############################################################################
# constraints parameters
############################################################################
epsilon = 7/tf.sqrt(1e2)
x_old = np.array([0.7006,0.1547,0.1440])
y_old = np.array([0.2993,0.8059,0.0297])
xc_divide_yc  = tf.convert_to_tensor(x_old/y_old,dtype=tf.float32)
yc_reciprocal = tf.convert_to_tensor(np.array([1,1,1])/y_old,dtype=tf.float32)
c_rgb = tf.convert_to_tensor(np.array([0.0114,0.0052,0.0427]),dtype=tf.float32)
c = c_rgb
multiply_factor = 1e3
for i_color in [0]:
    if (i_color == 0):
        g11=86e4/multiply_factor
        g12=-40e4/multiply_factor
        g22=45e4/multiply_factor
        x_desired = 0.313
        y_desired = 0.337
    elif i_color == 1:
        g11 = 76e4 / multiply_factor
        g12 = -35e4 / multiply_factor
        g22 = 38e4 / multiply_factor
        x_desired = 0.3287
        y_desired = 0.3417
    elif i_color == 2:
        g11 = 56e4 / multiply_factor
        g12 = -25e4 / multiply_factor
        g22 = 28e4 / multiply_factor
        x_desired = 0.346
        y_desired = 0.359
    elif i_color == 3:
        g11=39.5e4/multiply_factor
        g12=-21.5e4/multiply_factor
        g22=26e4/multiply_factor
        x_desired = 0.380
        y_desired = 0.380
    elif i_color == 4:
        g11=40e4/multiply_factor
        g12=-19.5e4/multiply_factor
        g22=28e4/multiply_factor
        x_desired = 0.459
        y_desired = 0.412
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
    # data1 = np.array(data1)

    # dd = tf.keras.backend.mean(data1, axis=1)
    # cc = tf.abs(tf.keras.backend.mean(data1, axis=0))
    # data1
    # print (data1)
    print (data1.shape)
    noise1 = np.random.normal(loc=0, scale=1, size=(N_train,n_channel))
    print (noise1.shape)
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
    # encoded_s_combine = Dense(M , activation='relu')(encoded_s_combine)
    # encoded_s_combine = Dense(M , activation='relu')(encoded_s_combine)
    encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
    # print (encoded_s_combine[1,:])
    # encoded_constraint = Dense(2 * n_channel, activation='relu')(encoded_s_combine)

    # ###########################################################
    # TX signal constraints
    # ###########################################################
    # print (encoded_s_combine[1,:])
    # encoded_s_combine = Dense(n_channel, activation='relu')(encoded_s_combine)
    # encoded_s_combine = BatchNormalization(momentum=0, center=False, scale=False)(encoded_s_combine)
    # softplus & hard_sigmoid & exponential (good)
    # relu (bad) linear (wrong)
    # encoded_constraint = Dense(n_channel, activation='elu')(encoded_s_combine) + 1
    encoded_s_combine = Dense(n_channel, activation='linear')(encoded_s_combine)
    encoded_constraint = tf.keras.activations.elu(encoded_s_combine,alpha=2.0) + 2
    # tf.keras.layers.ELU(alpha=1.0)
    # encoded_s_combine = Dense(n_channel, activation='tanh')(encoded_s_combine)
    # encoded_constraint = encoded_s_combine / tf.math.reduce_max(encoded_s_combine)
    # encoded_constraint = tf.keras.activations.softsign(encoded_s_combine) + 1
    # encoded_constraint = Dense(n_channel, activation='softmax')(encoded_s_combine)
    # encoded_constraint = Dense(n_channel, activation='relu')(encoded_s_combine)
    # encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,tf.math.maximum(encoded_s_combine,0.0)))(encoded_s_combine)
    # encoded_constraint = Lambda(lambda x: tf.math.minimum(1.0,encoded_s_combine))(encoded_s_combine)
    # print ("encoded_constraint",encoded_constraint[1,:])

    # total optical power (lm) constraint added to loss
    # total_o_power_train = 190
    # print ("encoded_constraint:",encoded_constraint.shape)
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
    SNR_train_dB_train_input = Input(shape=(1,))
    # SNR_train = 10. ** (SNR_train_dB_train_input / 10.)  # coverted 7 db of EbNo
    SNR_train = 10. ** (10 / 10.)  # coverted 7 db of EbNo
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
    rec_without_noise_normalize = (rec_without_noise ) / tf.sqrt(
                tf.reduce_mean(
                    (rec_without_noise - tf.reduce_mean(rec_without_noise)) ** 2
                )
            )
    # rec_without_noise_normalize = rec_without_noise
    noise_input = Input(shape=(n_channel,))

    noise_std_train = tf.sqrt(1/SNR_train)
    rec_with_noise = noise_input*noise_std_train*tf.sqrt(1+psi_2*rec_without_noise_normalize) + (rec_without_noise_normalize)
    # rec_with_noise = tf.concat([rec_with_noise_0,rec_with_noise_1,rec_with_noise_2],axis=1)

    # for i_n_channel in range(0,n_channel):
    #     rec_with_noise = GaussianNoise( )(rec_without_noise_normalize[:,])
    # rec_with_noise = GaussianNoise(
    #     tf.sqrt(1 / SNR_train))(rec_without_noise_normalize)
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
    autoencoder = Model(inputs=[input_signal,SNR_train_dB_train_input,noise_input], outputs=[decoded1_s1, color_loss])
    adam = Adam()  # SGD converge much slower than Adam
    # adam = SGD()  # SGD converge much slower than Adam


    # callback
    # class MyCallback(Callback):
    #     def __init__(self, alpha):
    #         self.alpha = alpha
            # self.beta  = beta
            # self.gamma = gamma

        # customize your behavior
        # def on_batch_end(self, epoch, logs={}):
            # results = [logs['bler_loss_loss'], logs['color_loss_loss']]
            # results = [1, 1]
            # K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
            # print("\n epoch %s, alpha = %s, 1-alpha = %s" % (
            #     epoch + 1, K.get_value(self.alpha), 1-K.get_value(self.alpha)))

        # def on_epoch_end(self, epoch, logs={}):
            # results = [logs['bler_loss_loss'], logs['color_loss_loss']]
            # K.set_value(self.alpha,  results[0] / (results[0] + results[1]) )
            # print("\n epoch %s, alpha = %s" % (
            #     epoch + 1, K.get_value(self.alpha)))

    autoencoder.compile(optimizer=adam,
                        loss={
                            'bler_loss': 'categorical_crossentropy',
                            'color_loss': lambda y_true, y_pred: y_pred},
                        loss_weights={
                            'bler_loss': alpha,
                            'color_loss': beta},
                        experimental_run_tf_function=False
                        )
    # printing summary of layers and it's trainable parameters
    print(autoencoder.summary())
    # ###########################################################
    # training auto encoder
    # ###########################################################
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=2, mode='min')
    reduce_learn_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1)
    random_const = np.random.randn(N_train, 1)
    # constant = tf.expand_dims(tf.convert_to_tensor(1),axis=0)
    autoencoder.fit([data1,SNR_train_dB_1,noise1], [data1, random_const],
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[reduce_learn_rate
                                ,early_stopping
                                # ,MyCallback(alpha)
                                ]
                    )

    ############################################################################
    # ### Make encoder and decoder
    ###########################################################################
    # making encoder from full autoencoder
    encoder_s1 = Model([input_signal,SNR_train_dB_train_input,noise_input], encoded_constraint)

    # making decoder from full autoencoder
    dec_input_1 = Input(shape=(n_channel,))
    # deco_s1 = autoencoder.layers[-14](dec_input_1)
    # deco_s1 = autoencoder.layers[-12](deco_s1)
    # deco_s1 = autoencoder.layers[-10](deco_s1)
    # deco_s1 = autoencoder.layers[-11](dec_input_1)
    deco_s1 = autoencoder.layers[-6](dec_input_1)
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
        scatter_point.append(encoder_s1.predict([np.expand_dims(temp, axis=0),SNR_train_dB_1,noise1]))
    scatter_point = np.array(scatter_point)
    scatter_point = scatter_point/np.max(scatter_point) # normalize to 1
    print(scatter_point.shape)
    print(scatter_point)
    print ("average current:",np.mean(scatter_point, axis=0)  )
    average_current = np.mean(scatter_point, axis=0) 
    x_mixed_final = tf.reduce_sum(xc_divide_yc*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)\
                    /tf.reduce_sum(yc_reciprocal*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)
    y_mixed_final = tf.reduce_sum(np.mean(scatter_point, axis=0)[0,:]/c,axis=0)\
                    /tf.reduce_sum(yc_reciprocal*np.mean(scatter_point, axis=0)[0,:]/c,axis=0)
    print("x_mixed_final:",x_mixed_final)
    print("y_mixed_final:",y_mixed_final)
    print("Macadam eclipse constraints:",g11 * tf.square(x_mixed_final - x_desired)
          + 2 * g12 * (x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(
        y_mixed_final - y_desired) )

    g11=g11*multiply_factor
    g12=g12*multiply_factor
    g22=g22*multiply_factor
    print("Macadam eclipse constraints (real):",g11 * tf.square(x_mixed_final - x_desired)
          + 2 * g12 * (x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(
        y_mixed_final - y_desired)  )

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
    plt.xlabel('Red',color='r')
    plt.ylabel('Green',color='g')
    ax.set_zlabel('Blue',color='b')
    for i_point in range(0,M):
        label = '(%.2f, %.2f, %.2f)' % (scatter_point[i_point,0,0],scatter_point[i_point,0,1],scatter_point[i_point,0,2])
        ax.text(scatter_point[i_point,0,0],scatter_point[i_point,0,1],scatter_point[i_point,0,2],label)
    # plt.show()


    ############################################################################
    # Autoencoder BLER(block error rate) performance
    ###########################################################################
    SNR_range_dB = np.arange(0, 30, 5)
    print("SNR_range_dB:", SNR_range_dB)

    bler_s1_sum = np.zeros(len(SNR_range_dB))
    for i_MonteCarlo in range(0, N_MonteCarlo):
        # ------------ generate one hot encoded vector for test-----------------
        test_label_s1 = np.random.randint(M, size=N_test)
        test_data1 = tf.one_hot(test_label_s1, M, on_value=1.0, off_value=0.0)
        nosie_normalize = np.random.normal(size=(N_test, n_channel))
        # --------------------------------------------------------
        bler_s1 = [None] * len(SNR_range_dB)
        for n in range(0, len(SNR_range_dB)):
            SNR = 10.0 ** (SNR_range_dB[n] / 10.0)
            noise_std = np.sqrt(1 / SNR)
            noise_mean = 0
            noise = noise_std * nosie_normalize  # * 0
            tx_out = encoder_s1.predict([test_data1,SNR_train_dB_1,noise1])
            tx_out = tx_out/tf.reduce_max(tx_out)
            test_r1_without_noise = tf.matmul(H,
                                              tx_out,
                                              transpose_a=False,
                                              transpose_b=True,
                                              adjoint_a=False,
                                              adjoint_b=False,
                                              a_is_sparse=False,
                                              b_is_sparse=False,
                                              name=None)
            test_r1_without_noise = tf.transpose(test_r1_without_noise)
            # normalize
            test_r1_without_noise = (test_r1_without_noise ) / tf.sqrt(
                tf.reduce_mean(
                    (test_r1_without_noise - tf.reduce_mean(test_r1_without_noise)) ** 2
                )
            )
            pred_final_signal_s1 = decoder_s1.predict(test_r1_without_noise + noise * tf.sqrt(1 + psi_2 * test_r1_without_noise))
            pred_output_s1 = tf.argmax(pred_final_signal_s1, axis=1)
            bler_s1[n] = sum(1. * (tf.cast(
                pred_output_s1 != tf.convert_to_tensor(test_label_s1, dtype=tf.int64), tf.float32
            ))) / N_test
        print("bler_s1:", bler_s1)
        bler_s1_sum = bler_s1_sum + bler_s1
    bler_s1_mean = bler_s1_sum / N_MonteCarlo
    # record the bler and eclipse constraints
    if(i_color == 0):
        bler_6500K = bler_s1_mean
        color_loss_6500K = g11 * tf.square(x_mixed_final - x_desired) + 2 * g12 * (
                x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(y_mixed_final - y_desired)
        average_current_6500K = average_current
        print("bler_6500K:", bler_6500K)
        print("\n")
    elif (i_color == 1):
        bler_5700K = bler_s1_mean
        color_loss_5700K = g11 * tf.square(x_mixed_final - x_desired) + 2 * g12 * (
                x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(y_mixed_final - y_desired)
        average_current_5700K = average_current
        print("bler_5700K:", bler_5700K)
        print("\n")
    elif (i_color == 2):
        bler_5000K = bler_s1_mean
        color_loss_5000K = g11 * tf.square(x_mixed_final - x_desired) + 2 * g12 * (
                x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(y_mixed_final - y_desired)
        average_current_5000K = average_current
        print("bler_5000K:", bler_5000K)
        print("\n")
    elif (i_color == 3):
        bler_4000K = bler_s1_mean
        color_loss_4000K = g11 * tf.square(x_mixed_final - x_desired) + 2 * g12 * (
                x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(y_mixed_final - y_desired)
        average_current_4000K = average_current
        print("bler_4000K:", bler_4000K)
        print("\n")
    elif (i_color == 4):
        bler_2700K = bler_s1_mean
        color_loss_2700K = g11 * tf.square(x_mixed_final - x_desired) + 2 * g12 * (
                x_mixed_final - x_desired) * (y_mixed_final - y_desired) + g22 * tf.square(y_mixed_final - y_desired)
        average_current_2700K = average_current
        print("bler_2700K:", bler_2700K)
        print("\n")

############################################################################
# performance visualization
###########################################################################
print("bler_6500K:", bler_6500K)
print("color_loss_6500K:", color_loss_6500K)
print("average_current_6500K:", average_current_6500K)
print("\n")

# print("bler_5000K:", bler_5000K)
# print("color_loss_5000K:", color_loss_5000K)
# print("average_current_5000K:", average_current_5000K)
# print("\n")

# print("bler_2700K:", bler_2700K)
# print("color_loss_2700K:", color_loss_2700K)
# print("average_current_2700K:", average_current_2700K)

# #### plot BLER curve
fig = plt.figure()
plt.plot(SNR_range_dB, bler_6500K, 'ro-',
         label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')
# plt.plot(SNR_range_dB, bler_5000K, 'go-',
#          label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')
# plt.plot(SNR_range_dB, bler_2700K, 'bo-',
#          label='Autoencoder(' + str(n_channel) + ',' + str(k) + ')_s1')

plt.yscale('log')
plt.xlabel('SNR Range')
plt.ylabel('Block Error Rate')
plt.grid(True)
plt.legend(loc='upper right', ncol=1)
plt.xticks(SNR_range_dB, SNR_range_dB[::1])
plt.show()
print(mpl.get_backend())
