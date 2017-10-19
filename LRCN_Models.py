# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:12:43 2017

@author: JMJ136
"""
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Conv3D
from CustomLayers import Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU, ELU
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

#%%
def LRCN_v1(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    # contracting
    lay_c1 = Conv2D(10, (3, 3),name='Conv_1')(lay_input)
    bn = BatchNormalization()(lay_c1)
    lay_c1a = ELU(name='elu_1')(bn)
    
    lay_c2 = Conv2D(20,(3,3),name='Conv_2')(lay_c1a)
    bn = BatchNormalization()(lay_c2)
    lay_c2a = ELU(name='elu_2')(bn)
    
    lay_c3 = Conv2D(30,(3,3),name='Conv_3')(lay_c2a)
    bn = BatchNormalization()(lay_c3)
    lay_c3a = ELU(name='elu_3')(bn)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2),name='Conv_4')(lay_c3a)
    bn = BatchNormalization()(lay_c4)
    lay_c4a = ELU(name='elu_4')(bn)
    
    lay_c5 = Conv2D(50,(3,3),name='Conv_5')(lay_c4a)
    bn = BatchNormalization()(lay_c5)
    lay_c5a = ELU(name='elu_5')(bn)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2),name='Conv_6')(lay_c5a)
    bn = BatchNormalization()(lay_c6)
    lay_c6a = ELU(name='elu_6')(bn)
    
    lay_c7 = Conv2D(70,(3,3),name='Conv_7')(lay_c6a)
    bn = BatchNormalization()(lay_c7)
    lay_c7a = ELU(name='elu_7')(bn)
    
    lay_c8 = Conv2D(80,(3,3),name='Conv_8')(lay_c7a)
    bn = BatchNormalization()(lay_c8)
    lay_c8a = ELU(name='elu_8')(bn)
    
    lay_c9 = Conv2D(90,(3,3),name='Conv_9')(lay_c8a)
    bn = BatchNormalization()(lay_c9)
    lay_c9a = ELU(name='elu_9')(bn)
    
    # expanding
    lay_d9 = Conv2DTranspose(80,(3,3),name='DeConv_9')(lay_c9a)
    bn = BatchNormalization()(lay_d9)
    lay_d9a = ELU(name='elu_d9')(bn)
    lay_cd9 = concatenate([lay_c8a,lay_d9a],name='concat_9')
    
    lay_d8 = Conv2DTranspose(70,(3,3),name='DeConv_8')(lay_cd9)
    bn = BatchNormalization()(lay_d8)
    lay_d8a = ELU(name='elu_d8')(bn)
    lay_cd8 = concatenate([lay_c7a,lay_d8a],name='concat_8')
    
    lay_d7 = Conv2DTranspose(60,(3,3),name='DeConv_7')(lay_cd8)
    bn = BatchNormalization()(lay_d7)
    lay_d7a = ELU(name='elu_d7')(bn)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv2DTranspose(50,(5,5),name='DeConv_6')(lay_cd7)
    bn = BatchNormalization()(lay_d6)
    lay_d6a = ELU(name='elu_d6')(bn)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv2DTranspose(50,(3,3),name='DeConv_5')(lay_cd6)
    bn = BatchNormalization()(lay_d5)
    lay_d5a = ELU(name='elu_d5')(bn)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv2DTranspose(40,(5,5),name='DeConv_4')(lay_cd5)
    bn = BatchNormalization()(lay_d4)
    lay_d4a = ELU(name='elu_d4')(bn)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv2DTranspose(30,(3,3),name='DeConv_3')(lay_cd4)
    bn = BatchNormalization()(lay_d3)
    lay_d3a = ELU(name='elu_d3')(bn)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv2DTranspose(20,(3,3),name='DeConv_2')(lay_cd3)
    bn = BatchNormalization()(lay_d2)
    lay_d2a = ELU(name='elu_d2')(bn)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv2DTranspose(10,(3,3),name='DeConv_1')(lay_cd2)
    bn = BatchNormalization()(lay_d1)
    lay_d1a = ELU(name='elu_d1')(bn)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)