# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:12:43 2017

@author: JMJ136
"""
from keras.layers import Input, Conv2D, concatenate, TimeDistributed, LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

#%%
def CompModel_v1(samp_input):
    # first version of image comparion model
    lay_input_1 = Input(shape=(samp_input.shape[1:]))
    lay_input_2 = Input(shape=(samp_input.shape[1:]))
    # input 1 contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu',name='conv1_1')(lay_input_1)
    bn1 = BatchNormalization()(lay_c1)
    lay_c2 = Conv2D(20,(3,3),activation='relu',name='conv1_2')(bn1)
    bn2 = BatchNormalization()(lay_c2)    
    lay_c3 = Conv2D(30,(3,3),activation='relu',name='conv1_3')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    lay_c4 = Conv2D(40,(3,3),strides=(2,2),activation='relu',name='conv1_4')(bn3)
    bn4 = BatchNormalization()(lay_c4)    
    lay_c5 = Conv2D(50,(3,3),activation='relu',name='conv1_5')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    lay_c6 = Conv2D(60,(3,3),activation='relu',name='conv1_6')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    lay_c7 = Conv2D(70,(3,3),activation='relu',name='conv1_7')(bn6)
    bn7 = BatchNormalization()(lay_c7)
    lay_c8 = Conv2D(80,(3,3),activation='relu',name='conv1_9')(bn7)
    contract_1 = BatchNormalization()(lay_c8)
    # input 2 contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu',name='conv2_1')(lay_input_2)
    bn1 = BatchNormalization()(lay_c1)
    lay_c2 = Conv2D(20,(3,3),activation='relu',name='conv2_2')(bn1)
    bn2 = BatchNormalization()(lay_c2)    
    lay_c3 = Conv2D(30,(3,3),activation='relu',name='conv2_3')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    lay_c4 = Conv2D(40,(3,3),strides=(2,2),activation='relu',name='conv2_4')(bn3)
    bn4 = BatchNormalization()(lay_c4)    
    lay_c5 = Conv2D(50,(3,3),activation='relu',name='conv2_5')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    lay_c6 = Conv2D(60,(3,3),activation='relu',name='conv2_6')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    lay_c7 = Conv2D(70,(3,3),activation='relu',name='conv2_7')(bn6)
    bn7 = BatchNormalization()(lay_c7)
    lay_c8 = Conv2D(80,(3,3),activation='relu',name='conv2_9')(bn7)
    contract_2 = BatchNormalization()(lay_c8)
    
    # concatenation
    concat1 = concatenate([contract_1,contract_2])
    
    #classification
    lay_conv = Conv2D(100,(3,3),activation='relu',name='conv1')(concat1)
    lay_conv = Conv2D(100,(3,3),activation='relu',name='conv2')(lay_conv)
    lay_bn = BatchNormalization()(lay_conv)
    lay_drop = Dropout(.5)(lay_bn)
    lay_conv = Conv2D(80,(5,5),strides=(2,2),activation='relu',name='conv3')(lay_drop)
    lay_conv = Conv2D(80,(3,3),activation='relu',name='conv4')(lay_conv)
    lay_bn = BatchNormalization()(lay_conv)
    lay_drop = Dropout(.5)(lay_bn)
    lay_conv = Conv2D(60,(5,5),strides=(2,2),activation='relu',name='conv5')(lay_drop)
    lay_conv = Conv2D(60,(3,3),activation='relu',name='conv6')(lay_conv)
    lay_bn = BatchNormalization()(lay_conv)
    lay_mp = MaxPooling2D(pool_size=(2,2))(lay_bn)
    lay_drop = Dropout(.5)(lay_mp)
    
    lay_flat = Flatten(name='Flatten')(lay_drop)
    lay_den = Dense(64,activation='relu',name='FullyConnected')(lay_flat)
    lay_drop = Dropout(.5)(lay_den)
    lay_out = Dense(1,activation='sigmoid',name='Classifier')(lay_drop)
    
    return Model(inputs=[lay_input_1,lay_input_2], outputs=lay_out)

#%%
def CompModel_v2(samp_input):
    # LRCN model
    inp_shape = samp_input.shape
    lay_input = Input(shape=(inp_shape[2:]))
    lay_conv = Conv2D(8, (3, 3))(lay_input)
    lay_conv = Conv2D(16, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_conv = Conv2D(32, (3, 3))(lay_mp)
    lay_conv = Conv2D(64, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_conv = Conv2D(64, (3, 3))(lay_mp)
    lay_conv = Conv2D(64, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_flat = Flatten()(lay_mp)
    cnn_model = Model(lay_input, lay_flat)
    lay_input2 = Input(shape=(inp_shape[1:]),name='sequence_input')
    lay_time = TimeDistributed(cnn_model)(lay_input2)  # the output will be a sequence of vectors
    lay_lstm = LSTM(64,dropout=.7,name='LSTM')(lay_time)  # the output will be a vector
    lay_class = Dense(1, activation='sigmoid',name='classifier')(lay_lstm)
    
    return Model(lay_input2,lay_class,name='LRCN_v1')
#%%
def CompModel_v3(samp_input):
    # multi-channel model
    inp_shape = samp_input.shape
    lay_input = Input(shape=(inp_shape[1:]))
    lay_conv = Conv2D(8, (3, 3))(lay_input)
    lay_conv = Conv2D(16, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_conv = Conv2D(32, (3, 3))(lay_mp)
    lay_conv = Conv2D(64, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_conv = Conv2D(64, (3, 3))(lay_mp)
    lay_conv = Conv2D(64, (3, 3))(lay_conv)
    lay_mp = MaxPooling2D((2, 2))(lay_conv)
    lay_flat = Flatten()(lay_mp)
    lay_dense = Dense(64,activation='relu')(lay_flat)
    lay_class = Dense(1, activation='sigmoid',name='classifier')(lay_dense)
    
    return Model(lay_input,lay_class)
#%%
def CompModel_v3(samp_input):
    # time-distributed sequential LRCN model
    inp_shape = samp_input.shape
    model = Sequential()
    # Convolutional Layers
    model.add(TimeDistributedConvolution2D(32, 3, 3, border_mode='same', input_shape=(number_of_timesteps, 1, height, width)))
    model.add(Activation('relu'))

    model.add(TimeDistributedMaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(TimeDistributedConvolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(TimeDistributedMaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(TimeDistributedFlatten())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # o = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    o = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)
    model.compile(loss="binary_crossentropy", optimizer=o)
    print "\t Model Created!"
    return model