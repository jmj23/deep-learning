# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:12:43 2017

@author: JMJ136
"""
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Lambda
from keras.layers.convolutional import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers.pooling import MaxPooling2D
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

#%%
def BatchModel_v1(samp_input):
    description = 'Simplest ResNet with ReLU activations'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3), activation='relu')(lay_input)
    
    lay_c2 = Conv2D(20,(3,3), activation='relu')(lay_c1)
    
    lay_c3 = Conv2D(30,(3,3), activation='relu')(lay_c2)
    
    lay_c4 = Conv2D(40,(3,3), activation='relu')(lay_c3)
    
    lay_c5 = Conv2D(50,(3,3), activation='relu')(lay_c4)
    
    lay_c6 = Conv2D(60,(3,3), activation='relu')(lay_c5)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3), activation='relu')(lay_c6)
    lay_cd6 = concatenate([lay_c5,lay_d6])
    
    lay_d5 = Conv2DTranspose(50,(3,3), activation='relu')(lay_cd6)
    lay_cd5 = concatenate([lay_c4,lay_d5])
    
    lay_d4 = Conv2DTranspose(40,(3,3), activation='relu')(lay_cd5)
    lay_cd4 = concatenate([lay_c3,lay_d4])
    
    lay_d3 = Conv2DTranspose(30,(3,3), activation='relu')(lay_cd4)
    lay_cd3 = concatenate([lay_c2,lay_d3])
    
    lay_d2 = Conv2DTranspose(20,(3,3), activation='relu')(lay_cd3)
    lay_cd2 = concatenate([lay_c1,lay_d2])
    
    lay_d1 = Conv2DTranspose(10,(3,3), activation='relu')(lay_cd2)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v2(samp_input):
    description = 'Sequential Model with ReLU activations'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3), activation='relu')(lay_input)
    
    lay_c2 = Conv2D(20,(3,3), activation='relu')(lay_c1)
    
    lay_c3 = Conv2D(30,(3,3), activation='relu')(lay_c2)
    
    lay_c4 = Conv2D(40,(3,3), activation='relu')(lay_c3)
    
    lay_c5 = Conv2D(50,(3,3), activation='relu')(lay_c4)
    
    lay_c6 = Conv2D(60,(3,3), activation='relu')(lay_c5)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3), activation='relu')(lay_c6)
    
    lay_d5 = Conv2DTranspose(50,(3,3), activation='relu')(lay_d6)
    
    lay_d4 = Conv2DTranspose(40,(3,3), activation='relu')(lay_d5)
    
    lay_d3 = Conv2DTranspose(30,(3,3), activation='relu')(lay_d4)
    
    lay_d2 = Conv2DTranspose(20,(3,3), activation='relu')(lay_d3)
    
    lay_d1 = Conv2DTranspose(10,(3,3), activation='relu')(lay_d2)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v3(samp_input):
    description = 'ResNet with ELU activations'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v4(samp_input):
    description = 'ResNet with PReLU activations'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = PReLU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3))(lay_c1a)
    lay_c2a = PReLU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = PReLU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3))(lay_c3a)
    lay_c4a = PReLU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = PReLU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3))(lay_c5a)
    lay_c6a = PReLU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(lay_c6a)
    lay_d6a = PReLU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = PReLU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = PReLU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = PReLU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = PReLU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = PReLU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v5(samp_input):
    description = 'ResNet with ELU activations and Batch Normalization on contracting side'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([bn5,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([bn4,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([bn3,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([bn2,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([bn1,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v6(samp_input):
    description = 'U-net duplicate'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1a = Conv2D(64, (3, 3),activation='relu')(lay_input)
    lay_c1b = Conv2D(64, (3, 3),activation='relu')(lay_c1a)
    lay_mp1 = MaxPooling2D(pool_size=(2, 2))(lay_c1b)
    
    lay_c2a = Conv2D(128, (3, 3),activation='relu')(lay_mp1)
    lay_c2b = Conv2D(128, (3, 3),activation='relu')(lay_c2a)
    lay_c2c = Conv2D(128, (3, 3),activation='relu')(lay_c2b)
    lay_mp2 = MaxPooling2D(pool_size=(2, 2))(lay_c2c)
    
    lay_c3a = Conv2D(256, (3, 3),activation='relu')(lay_mp2)
    lay_c3b = Conv2D(256, (3, 3),activation='relu')(lay_c3a)
    lay_mp3 = MaxPooling2D(pool_size=(2, 2))(lay_c3b)
    
    lay_c4a = Conv2D(512, (3, 3),activation='relu')(lay_mp3)
    lay_c4b = Conv2D(512, (3, 3),activation='relu')(lay_c4a)
    lay_mp4 = MaxPooling2D(pool_size=(2, 2))(lay_c4b)
    
    lay_c5a = Conv2D(1024,(3,3),activation='relu')(lay_mp4)
    lay_c5b = Conv2D(1024,(3,3),activation='relu')(lay_c5a)
    
    #expanding
    lay_up5 = Conv2DTranspose(512,(2,2),strides=(2,2),activation='relu')(lay_c5b)
    lay_crop4 = Cropping2D(cropping=4)(lay_c4b)
    lay_concat5 = concatenate([lay_up5,lay_crop4])
    lay_d4a = Conv2D(512, (3, 3),activation='relu')(lay_concat5)
    lay_d4b = Conv2D(512, (3, 3),activation='relu')(lay_d4a)
    
    lay_up4 = Conv2DTranspose(256,(2,2),strides=(2,2),activation='relu')(lay_d4b)
    lay_crop3 = Cropping2D(cropping=16)(lay_c3b)
    lay_concat4 = concatenate([lay_up4,lay_crop3])
    lay_d3a = Conv2D(256, (3, 3),activation='relu')(lay_concat4)
    lay_d3b = Conv2D(256, (3, 3),activation='relu')(lay_d3a)
    
    lay_up3 = Conv2DTranspose(128,(2,2),strides=(2,2),activation='relu')(lay_d3b)
    lay_crop2 = Cropping2D(cropping=41)(lay_c2b)
    lay_concat3 = concatenate([lay_up3,lay_crop2])
    lay_d2a = Conv2D(128, (3, 3),activation='relu')(lay_concat3)
    lay_d2b = Conv2D(128, (3, 3),activation='relu')(lay_d2a)
    
    lay_up2 = Conv2DTranspose(64,(2,2),strides=(2,2),activation='relu')(lay_d2b)
    lay_crop1 = Cropping2D(cropping=90)(lay_c1b)
    lay_concat2 = concatenate([lay_up2,lay_crop1])
    lay_d1a = Conv2D(64, (3, 3),activation='relu')(lay_concat2)
    lay_d1b = Conv2D(64, (3, 3),activation='relu')(lay_d1a)
    lay_d1c = Conv2D(64, (3, 3),activation='relu')(lay_d1b)
    lay_d1d = Conv2D(64, (3, 3),activation='relu')(lay_d1c)
    lay_stride1 = Conv2DTranspose(64,(3,3),strides=(2,2),activation='relu')(lay_d1d)   
    lay_stride2 = Conv2DTranspose(64,(3,3),strides=(2,2),activation='relu')(lay_stride1)
    
    # classifier
    lay_class = Conv2D(1,(1,1), activation='sigmoid')(lay_stride2)
    return Model(lay_input,lay_class), description
#%%
def BatchModel_v7(samp_input):
    description = 'ResNet with ELU activations and dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),dilation_rate=(2,2))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(5,5))(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(5,5))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v8(samp_input):
    description = 'ResNet with ELU activations and Batch Normalization on both sides'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v9(samp_input):
    description = 'ResNet with ELU activations and pooling/strided deconvolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    mp1 = MaxPooling2D(pool_size=(2,2))(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(mp1)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(6,6),strides=(2,2))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v10(samp_input):
    description = 'ResNet with ELU activations, dilated convolutions, and exponential filters'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(8, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(16,(3,3),dilation_rate=(2,2))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(32,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(64,(3,3),dilation_rate=(2,2))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(128,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(256,(3,3),dilation_rate=(2,2))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(256,(5,5))(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(128,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(64,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(32,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(16,(5,5))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(8,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v11(samp_input):
    description = 'ResNet with ELU activations, dilated convolutions, and complimentary filters'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(8, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(16,(3,3),dilation_rate=(2,2))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(32,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(64,(3,3),dilation_rate=(2,2))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(128,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(256,(3,3),dilation_rate=(2,2))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(128,(5,5))(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(192,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(224,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(240,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(248,(5,5))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(246,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v12(samp_input):
    description = 'ResNet with ELU activations, 2 extra layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3))(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3))(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3))(lay_c8a)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a])
    
    lay_d6 = Conv2DTranspose(60,(3,3))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v13(samp_input):
    description = 'ResNet with ELU, every other dilated convolutions, and 2 extra layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),dilation_rate=(2,2))(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2))(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2))(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3))(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),dilation_rate=(2,2))(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(5,5))(lay_c8a)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a])
    
    lay_d6 = Conv2DTranspose(50,(5,5))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(5,5))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v14(samp_input):
    description = 'ResNet with ELU, full BN, and 2 extra layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    lay_c7 = Conv2D(70,(3,3))(bn6)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    bn7 = BatchNormalization()(lay_c7a)
    
    lay_c8 = Conv2D(80,(3,3))(bn7)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    bn8 = BatchNormalization()(lay_c8a)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3))(bn8)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    bnd8 = BatchNormalization()(lay_d8a)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    bnd7 = BatchNormalization()(lay_d7a)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(3,3))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v15(samp_input):
    description = 'ResNet with ELU, full BN, and last 2 dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3),dilation_rate=(2,2))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(5,5))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(5,5))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v16(samp_input):
    description = 'ResNet with ELU, full BN, and first 2 dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),dilation_rate=(2,2))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3),dilation_rate=(2,2))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(5,5))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(5,5))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v17(samp_input):
    description = 'ResNet with ELU, full BN, Input concatenation'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    lay_cd1 = concatenate([lay_input,bnd1])
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_cd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v18(samp_input):
    description = 'ResNet with ReLU, full BN'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu')(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3),activation='relu')(bn6)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v19(samp_input):
    description = 'Inception blocks'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # block 1
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(lay_input)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(lay_input)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(lay_input)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    # block 2
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(mp)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    # block 3
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(mp)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    # block 4
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(mp)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    # block 5
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(mp)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    # block 6
    tower_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_1)
    tower_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(mp)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(mp)
    tower_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(tower_3)
    concat = concatenate([tower_1, tower_2, tower_3], axis=-1)
    mp = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat)
    
    lay_out = Conv2D(1,(1,1),activation='sigmoid')(mp)

    return Model(lay_input,lay_out), description

#%%
def BatchModel_v20(samp_input):
    description = 'ResNet with ReLU, full BN and 2 extra layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu')(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3),activation='relu')(bn6)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v21(samp_input):
    description = 'ResNet with ELU, full BN, and middle 2 dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3),dilation_rate=(2,2))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3))(bn6)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(5,5))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v22(samp_input):
    description = 'ResNet with ReLU, full BN, and middle 2 dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu',dilation_rate=(2,2))(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu',dilation_rate=(2,2))(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3),activation='relu')(bn6)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(5,5),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(5,5),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output')(bnd1)
    
    return Model(lay_input,lay_out), description

#%%
def BatchModel_v23(samp_input):
    description = 'ResNet with ELU, full BN, 8 layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    lay_c7 = Conv2D(70,(3,3))(bn6)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    bn7 = BatchNormalization()(lay_c7a)
    
    lay_c8 = Conv2D(80,(3,3))(bn7)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    bn8 = BatchNormalization()(lay_c8a)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3))(bn8)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    bnd8 = BatchNormalization()(lay_d8a)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    bnd7 = BatchNormalization()(lay_d7a)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(3,3))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v24(samp_input):
    description = 'ResNet with ReLU, full BN, 8 layers'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu')(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3),activation='relu')(bn6)
    bn7 = BatchNormalization()(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),activation='relu')(bn7)
    bn8 = BatchNormalization()(lay_c8)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3),activation='relu')(bn8)
    bnd8 = BatchNormalization()(lay_d8)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3),activation='relu')(lay_cd8)
    bnd7 = BatchNormalization()(lay_d7)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(3,3),activation='relu')(lay_cd7)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v25(samp_input):
    description = 'ResNet with ReLU, full BN, 8 layers, 2 middle dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu',dilation_rate=(2,2))(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu',dilation_rate=(2,2))(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3),activation='relu')(bn6)
    bn7 = BatchNormalization()(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),activation='relu')(bn7)
    bn8 = BatchNormalization()(lay_c8)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3),activation='relu')(bn8)
    bnd8 = BatchNormalization()(lay_d8)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3),activation='relu')(lay_cd8)
    bnd7 = BatchNormalization()(lay_d7)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(3,3),activation='relu')(lay_cd7)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(5,5),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(5,5),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v26(samp_input):
    description = 'ResNet with ELU, full BN, 8 layers, 2 middle dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3),dilation_rate=(2,2))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    lay_c7 = Conv2D(70,(3,3))(bn6)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    bn7 = BatchNormalization()(lay_c7a)
    
    lay_c8 = Conv2D(80,(3,3))(bn7)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    bn8 = BatchNormalization()(lay_c8a)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3))(bn8)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    bnd8 = BatchNormalization()(lay_d8a)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    bnd7 = BatchNormalization()(lay_d7a)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(3,3))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(5,5))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v27(samp_input):
    description = 'ResNet with ELU, full BN, 8 layers, 2 separated dilated convolutions'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    bn1 = BatchNormalization()(lay_c1a)
    
    lay_c2 = Conv2D(20,(3,3))(bn1)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    bn2 = BatchNormalization()(lay_c2a)
    
    lay_c3 = Conv2D(30,(3,3))(bn2)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    bn3 = BatchNormalization()(lay_c3a)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2))(bn3)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    bn4 = BatchNormalization()(lay_c4a)
    
    lay_c5 = Conv2D(50,(3,3))(bn4)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    bn5 = BatchNormalization()(lay_c5a)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2))(bn5)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    bn6 = BatchNormalization()(lay_c6a)
    
    lay_c7 = Conv2D(70,(3,3))(bn6)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    bn7 = BatchNormalization()(lay_c7a)
    
    lay_c8 = Conv2D(80,(3,3))(bn7)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    bn8 = BatchNormalization()(lay_c8a)
    
    # expanding
    lay_d8 = Conv2DTranspose(80,(3,3))(bn8)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    bnd8 = BatchNormalization()(lay_d8a)
    lay_cd8 = concatenate([bn7,bnd8])
    
    lay_d7 = Conv2DTranspose(70,(3,3))(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    bnd7 = BatchNormalization()(lay_d7a)
    lay_cd7 = concatenate([bn6,bnd7])
    
    lay_d6 = Conv2DTranspose(60,(5,5))(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    bnd6 = BatchNormalization()(lay_d6a)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    bnd5 = BatchNormalization()(lay_d5a)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(5,5))(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    bnd4 = BatchNormalization()(lay_d4a)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    bnd3 = BatchNormalization()(lay_d3a)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    bnd2 = BatchNormalization()(lay_d2a)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    bnd1 = BatchNormalization()(lay_d1a)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(bnd1)
    
    return Model(lay_input,lay_out), description
#%%
def BatchModel_v28(samp_input):
    description = 'ResNet with ReLU, full BN, and auxiliary output'
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3),activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),activation='relu')(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    aux_output = Conv2D(1,(1,1),activation='sigmoid')(bn6)
    aux_output_rs = Lambda(lambda image: K.tf.image.resize_images(image, (256, 256)),
                        name='aux_output')(aux_output)
    
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3),activation='relu')(bn6)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3),activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3),activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3),activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3),activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3),activation='relu')(lay_cd2)
    bnd1 = BatchNormalization()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='main_output')(bnd1)
    
    return Model(inputs=lay_input, outputs=[lay_out, aux_output_rs]), description