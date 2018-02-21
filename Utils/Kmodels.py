# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:12:43 2017

@author: JMJ136
"""
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, Conv3D
from keras.layers.convolutional import ZeroPadding2D, ZeroPadding3D, Cropping2D, Cropping3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers import Add
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.layers import Conv3DTranspose

#%%
def ResModel(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3), activation='relu')(lay_input)
    bn1 = BatchNormalization()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3), activation='relu')(bn1)
    bn2 = BatchNormalization()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3), activation='relu')(bn2)
    bn3 = BatchNormalization()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3), activation='relu')(bn3)
    bn4 = BatchNormalization()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3), activation='relu')(bn4)
    bn5 = BatchNormalization()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3), activation='relu')(bn5)
    bn6 = BatchNormalization()(lay_c6)
    
    # expanding
    lay_d6 = Conv2DTranspose(50,(3,3), activation='relu')(bn6)
    bnd6 = BatchNormalization()(lay_d6)
    lay_cd6 = concatenate([bn5,bnd6])
    
    lay_d5 = Conv2DTranspose(50,(3,3), activation='relu')(lay_cd6)
    bnd5 = BatchNormalization()(lay_d5)
    lay_cd5 = concatenate([bn4,bnd5])
    
    lay_d4 = Conv2DTranspose(40,(3,3), activation='relu')(lay_cd5)
    bnd4 = BatchNormalization()(lay_d4)
    lay_cd4 = concatenate([bn3,bnd4])
    
    lay_d3 = Conv2DTranspose(30,(3,3), activation='relu')(lay_cd4)
    bnd3 = BatchNormalization()(lay_d3)
    lay_cd3 = concatenate([bn2,bnd3])
    
    lay_d2 = Conv2DTranspose(20,(3,3), activation='relu')(lay_cd3)
    bnd2 = BatchNormalization()(lay_d2)
    lay_cd2 = concatenate([bn1,bnd2])
    
    lay_d1 = Conv2DTranspose(10,(3,3), activation='relu')(lay_cd2)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1)
    
    return Model(lay_input,lay_out)
#%%
def ResModel_noBN(samp_input):
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
    
    return Model(lay_input,lay_out)

#%%
def ResModel_noBN_prelu(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]))
    # contracting
    lay_c1 = Conv2D(10, (3, 3))(lay_input)
    lay_c1a = PReLU()(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3))(lay_c1a)
    lay_c2a = PReLU()(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3))(lay_c2a)
    lay_c3a = PReLU()(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3))(lay_c3a)
    lay_c4a = PReLU()(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3))(lay_c4a)
    lay_c5a = PReLU()(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3))(lay_c5a)
    lay_c6a = PReLU()(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3))(lay_c6a)
    lay_c7a = PReLU()(lay_c7)
    
    # expanding
    lay_d7 = Conv2DTranspose(60,(3,3))(lay_c7a)
    lay_d7a = PReLU()(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a])
    
    lay_d6 = Conv2DTranspose(50,(3,3))(lay_cd7)
    lay_d6a = PReLU()(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a])
    
    lay_d5 = Conv2DTranspose(50,(3,3))(lay_cd6)
    lay_d5a = PReLU()(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a])
    
    lay_d4 = Conv2DTranspose(40,(3,3))(lay_cd5)
    lay_d4a = PReLU()(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a])
    
    lay_d3 = Conv2DTranspose(30,(3,3))(lay_cd4)
    lay_d3a = PReLU()(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a])
    
    lay_d2 = Conv2DTranspose(20,(3,3))(lay_cd3)
    lay_d2a = PReLU()(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a])
    
    lay_d1 = Conv2DTranspose(10,(3,3))(lay_cd2)
    lay_d1a = PReLU()(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid')(lay_d1a)
    
    return Model(lay_input,lay_out)
#%%
def ResModel_noBN_elu(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    # contracting
    lay_c1 = Conv2D(10, (3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3),name='Conv_7')(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),name='Conv_8')(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    # expanding
    lay_d8 = Conv2DTranspose(70,(3,3),name='DeConv_8')(lay_c8a)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a],name='concat_8')
    
    lay_d7 = Conv2DTranspose(60,(3,3),name='DeConv_7')(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv2DTranspose(50,(3,3),name='DeConv_6')(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv2DTranspose(50,(3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv2DTranspose(40,(3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv2DTranspose(30,(3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv2DTranspose(20,(3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv2DTranspose(10,(3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)

#%%
def SeqModel(samp_input):
    model = Sequential()
    # Convolutions
    model.add(Conv2D(8 , (3, 3), input_shape=samp_input.shape[1:],
                               activation='relu',name='lay_c1'))
    model.add(BatchNormalization(name='bn_c1'))
    
    model.add(Conv2D(20,(3,3), activation='relu',name='lay_c2'))
    model.add(BatchNormalization(name='bn_c2'))
    
    model.add(Conv2D(30,(3,3), activation='relu',name='lay_c3'))
    model.add(BatchNormalization(name='bn_c3'))
    
    model.add(Conv2D(40,(3,3), activation='relu',name='lay_c4'))
    model.add(BatchNormalization(name='bn_c4'))
    
    model.add(Conv2D(50,(3,3), activation='relu',name='lay_c5'))
    model.add(BatchNormalization(name='bn_c5'))
    
    model.add(Conv2D(60,(3,3), activation='relu',name='lay_c6'))
    model.add(BatchNormalization(name='bn_c6'))
    
    # expanding]
    model.add(Conv2DTranspose(50,(3,3), activation='relu',name='lay_d6'))
    model.add(BatchNormalization(name='bn_d6'))
    
    model.add(Conv2DTranspose(50,(3,3), activation='relu',name='lay_d5'))
    model.add(BatchNormalization(name='bn_d5'))
    
    model.add(Conv2DTranspose(40,(3,3), activation='relu',name='lay_d4'))
    model.add(BatchNormalization(name='bn_d4'))
    
    model.add(Conv2DTranspose(30,(3,3), activation='relu',name='lay_d3'))
    model.add(BatchNormalization(name='bn_d3'))
    
    model.add(Conv2DTranspose(20,(3,3), activation='relu',name='lay_d2'))
    model.add(BatchNormalization(name='bn_d2'))
    
    model.add(Conv2DTranspose(10,(3,3), activation='relu',name='lay_d1'))
    
    # classifier
    model.add(Conv2D(1,(1,1), activation='sigmoid',name='lay_out'))
    
    return model

#%%
def SeqModel_noBN(samp_input):
    model = Sequential()
    # Convolutions
    model.add(Conv2D(8 , (3, 3), input_shape=samp_input.shape[1:],
                               activation='relu',name='lay_c1'))
    
    model.add(Conv2D(20,(3,3), activation='relu',name='lay_c2'))
    
    model.add(Conv2D(30,(3,3), activation='relu',name='lay_c3'))
    
    model.add(Conv2D(40,(3,3), activation='relu',name='lay_c4'))
    
    model.add(Conv2D(50,(3,3), activation='relu',name='lay_c5'))
    
    model.add(Conv2D(60,(3,3), activation='relu',name='lay_c6'))
    
    # expanding]
    model.add(Conv2DTranspose(50,(3,3), activation='relu',name='lay_d6'))
    
    model.add(Conv2DTranspose(50,(3,3), activation='relu',name='lay_d5'))
    
    model.add(Conv2DTranspose(40,(3,3), activation='relu',name='lay_d4'))
    
    model.add(Conv2DTranspose(30,(3,3), activation='relu',name='lay_d3'))
    
    model.add(Conv2DTranspose(20,(3,3), activation='relu',name='lay_d2'))
    
    model.add(Conv2DTranspose(10,(3,3), activation='relu',name='lay_d1'))
    
    # classifier
    model.add(Conv2D(1,(1,1), activation='sigmoid',name='lay_out'))
    
    return model
#%%
def ResModel_3D_elu(samp_input):
    lay_input = Input(shape=(None,samp_input.shape[2],samp_input.shape[3],samp_input.shape[4]),
                             name='input_layer')
    # contracting
    lay_c1 = Conv3D(10, (3, 3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv3D(20,(3, 3, 3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv3D(30,(3, 3, 3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv3D(40,(3, 3, 3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv3D(50,(3, 3, 3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv3D(60,(3, 3, 3),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv3D(70,(3, 3, 3),name='Conv_7')(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv3D(80,(3, 3, 3),name='Conv_8')(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    # expanding
    lay_d8 = Conv3DTranspose(70,(3,3,3),name='DeConv_8')(lay_c8a)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a],name='concat_8')
    
    lay_d7 = Conv3DTranspose(60,(3,3,3),name='DeConv_7')(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv3DTranspose(50,(3,3,3),name='DeConv_6')(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv3DTranspose(50,(3,3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv3DTranspose(40,(3,3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv3DTranspose(30,(3,3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv3DTranspose(20,(3,3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv3DTranspose(10,(3,3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)

#%%
def ResModel_3D_elu_small(samp_input):
    lay_input = Input(shape=(None,samp_input.shape[2],samp_input.shape[3],samp_input.shape[4]),
                             name='input_layer')
    # contracting
    lay_c1 = Conv3D(10, (3, 3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv3D(20,(3, 3, 3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv3D(30,(3, 3, 3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv3D(40,(3, 3, 3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv3D(50,(3, 3, 3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv3D(60,(3, 3, 3),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv3D(70,(3, 3, 3),name='Conv_7')(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    # expanding    
    lay_d7 = Conv3DTranspose(60,(3,3,3),name='DeConv_7')(lay_c7a)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv3DTranspose(50,(3,3,3),name='DeConv_6')(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv3DTranspose(50,(3,3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv3DTranspose(40,(3,3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv3DTranspose(30,(3,3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv3DTranspose(20,(3,3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv3DTranspose(10,(3,3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)

#%%
def ResModel_3D_v1(samp_input):
    lay_input = Input(shape=(None,samp_input.shape[2],samp_input.shape[3],samp_input.shape[4]),
                             name='input_layer')
    # contracting
    lay_c1 = Conv3D(10, (3, 3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv3D(20,(3, 3, 3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv3D(30,(3, 3, 3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv3D(40,(3, 3, 3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv3D(50,(3, 3, 3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)

    # expanding        
    lay_d5 = Conv3DTranspose(50,(3,3,3),name='DeConv_5')(lay_c5a)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv3DTranspose(40,(3,3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv3DTranspose(30,(3,3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv3DTranspose(20,(3,3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv3DTranspose(10,(3,3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid',name='output_layer')(lay_d1a)

    return Model(lay_input,lay_out)
#%%
def ResModel_3D_v2(samp_input):
    lay_input = Input(shape=(None,None,samp_input.shape[3],samp_input.shape[4]),
                             name='input_layer')
    # contracting
    lay_c1 = Conv3D(10, (3, 3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv3D(20,(3, 3, 3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv3D(30,(3, 3, 3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv3D(40,(3, 3, 3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv3D(50,(3, 3, 3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv3D(60,(3, 3, 3),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    # expanding
    lay_d6 = Conv3DTranspose(60,(3,3,3),name='DeConv_6')(lay_c6a)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv3DTranspose(50,(3,3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv3DTranspose(40,(3,3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv3DTranspose(30,(3,3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv3DTranspose(20,(3,3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv3DTranspose(10,(3,3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid',name='output_layer')(lay_d1a)

    return Model(lay_input,lay_out)

#%%
def ResModel_v7(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    # contracting
    lay_c1 = Conv2D(10, (3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3),name='Conv_7')(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),name='Conv_8')(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    lay_c9 = Conv2D(90,(3,3),name='Conv_9')(lay_c8a)
    lay_c9a = ELU(name='elu_9')(lay_c9)
    
    lay_c10 = Conv2D(100,(3,3),name='Conv_10')(lay_c9a)
    lay_c10a = ELU(name='elu_10')(lay_c10)
    
    # expanding
    lay_d10 = Conv2DTranspose(90,(3,3),name='DeConv_10')(lay_c10a)
    lay_d10a = ELU(name='elu_d10')(lay_d10)
    lay_cd10 = concatenate([lay_c9a,lay_d10a],name='concat_10')
    
    lay_d9 = Conv2DTranspose(80,(3,3),name='DeConv_9')(lay_cd10)
    lay_d9a = ELU(name='elu_d9')(lay_d9)
    lay_cd9 = concatenate([lay_c8a,lay_d9a],name='concat_9')
    
    lay_d8 = Conv2DTranspose(70,(3,3),name='DeConv_8')(lay_cd9)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a],name='concat_8')
    
    lay_d7 = Conv2DTranspose(60,(3,3),name='DeConv_7')(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv2DTranspose(50,(3,3),name='DeConv_6')(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv2DTranspose(50,(3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv2DTranspose(40,(3,3),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv2DTranspose(30,(3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv2DTranspose(20,(3,3),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv2DTranspose(10,(3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)

#%%
def ResModel_v8(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    # contracting
    lay_c1 = Conv2D(10, (3, 3),name='Conv_1')(lay_input)
    lay_c1a = ELU(name='elu_1')(lay_c1)
    
    lay_c2 = Conv2D(20,(3,3),dilation_rate=(2,2),name='Conv_2')(lay_c1a)
    lay_c2a = ELU(name='elu_2')(lay_c2)
    
    lay_c3 = Conv2D(30,(3,3),name='Conv_3')(lay_c2a)
    lay_c3a = ELU(name='elu_3')(lay_c3)
    
    lay_c4 = Conv2D(40,(3,3),dilation_rate=(2,2),name='Conv_4')(lay_c3a)
    lay_c4a = ELU(name='elu_4')(lay_c4)
    
    lay_c5 = Conv2D(50,(3,3),name='Conv_5')(lay_c4a)
    lay_c5a = ELU(name='elu_5')(lay_c5)
    
    lay_c6 = Conv2D(60,(3,3),dilation_rate=(2,2),name='Conv_6')(lay_c5a)
    lay_c6a = ELU(name='elu_6')(lay_c6)
    
    lay_c7 = Conv2D(70,(3,3),name='Conv_7')(lay_c6a)
    lay_c7a = ELU(name='elu_7')(lay_c7)
    
    lay_c8 = Conv2D(80,(3,3),dilation_rate=(2,2),name='Conv_8')(lay_c7a)
    lay_c8a = ELU(name='elu_8')(lay_c8)
    
    lay_c9 = Conv2D(90,(3,3),name='Conv_9')(lay_c8a)
    lay_c9a = ELU(name='elu_9')(lay_c9)
    
    # expanding
    lay_d9 = Conv2DTranspose(80,(3,3),name='DeConv_9')(lay_c9a)
    lay_d9a = ELU(name='elu_d9')(lay_d9)
    lay_cd9 = concatenate([lay_c8a,lay_d9a],name='concat_9')
    
    lay_d8 = Conv2DTranspose(70,(5,5),name='DeConv_8')(lay_cd9)
    lay_d8a = ELU(name='elu_d8')(lay_d8)
    lay_cd8 = concatenate([lay_c7a,lay_d8a],name='concat_8')
    
    lay_d7 = Conv2DTranspose(60,(3,3),name='DeConv_7')(lay_cd8)
    lay_d7a = ELU(name='elu_d7')(lay_d7)
    lay_cd7 = concatenate([lay_c6a,lay_d7a],name='concat_7')
    
    lay_d6 = Conv2DTranspose(50,(5,5),name='DeConv_6')(lay_cd7)
    lay_d6a = ELU(name='elu_d6')(lay_d6)
    lay_cd6 = concatenate([lay_c5a,lay_d6a],name='concat_6')
    
    lay_d5 = Conv2DTranspose(50,(3,3),name='DeConv_5')(lay_cd6)
    lay_d5a = ELU(name='elu_d5')(lay_d5)
    lay_cd5 = concatenate([lay_c4a,lay_d5a],name='concat_5')
    
    lay_d4 = Conv2DTranspose(40,(5,5),name='DeConv_4')(lay_cd5)
    lay_d4a = ELU(name='elu_d4')(lay_d4)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv2DTranspose(30,(3,3),name='DeConv_3')(lay_cd4)
    lay_d3a = ELU(name='elu_d3')(lay_d3)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv2DTranspose(20,(5,5),name='DeConv_2')(lay_cd3)
    lay_d2a = ELU(name='elu_d2')(lay_d2)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv2DTranspose(10,(3,3),name='DeConv_1')(lay_cd2)
    lay_d1a = ELU(name='elu_d1')(lay_d1)
    
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_d1a)
    
    return Model(lay_input,lay_out)

#%%
def ResModel_v9(samp_input):
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
#%%
def BlockModel_v1(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    
    padamt = 1
    crop = Cropping2D(cropping=((0, padamt), (0, padamt)), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(10*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(10*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(10*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(10*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(10*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(10*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-4    
    for rr in range(2,5):
        lay_conv1 = Conv2D(10*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(10*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(10*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(10*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(10*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(10*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 4
    dd=4
    lay_deconv1 = Conv2D(10*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(10*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(10*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(10*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(10*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_stride = Conv2DTranspose(10*dd,(3,3),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
        
    # expanding blocks 3-1
    expnums = list(range(1,4))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(10*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(10*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(10*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(10*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(10*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_stride = Conv2DTranspose(10*dd,(3,3),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_act)
    
    zeropad = ZeroPadding2D(padding=((0,padamt), (0, padamt)), data_format=None)(lay_out)
    
    return Model(lay_input,zeropad)
#%%
def ResModel_3D_v3(samp_input):
    lay_input = Input(shape=(None,None,samp_input.shape[3],samp_input.shape[4]),
                             name='input_layer')
    # contracting
    lay_c1 = Conv3D(10, (3, 3, 3),name='Conv_1')(lay_input)
    lay_bn = BatchNormalization()(lay_c1)
    lay_c1a = ELU(name='elu_1')(lay_bn)
    
    lay_c2 = Conv3D(20,(3, 3, 3),name='Conv_2')(lay_c1a)
    lay_bn = BatchNormalization()(lay_c2)
    lay_c2a = ELU(name='elu_2')(lay_bn)
    
    lay_c3 = Conv3D(30,(3, 3, 3),name='Conv_3')(lay_c2a)
    lay_bn = BatchNormalization()(lay_c3)
    lay_c3a = ELU(name='elu_3')(lay_bn)
    
    lay_c4 = Conv3D(40,(3, 3, 3),name='Conv_4')(lay_c3a)
    lay_bn = BatchNormalization()(lay_c4)
    lay_c4a = ELU(name='elu_4')(lay_bn)
    
    # expanding     
    lay_d4 = Conv3DTranspose(40,(3,3,3),name='DeConv_4')(lay_c4a)
    lay_bn = BatchNormalization()(lay_d4)
    lay_d4a = ELU(name='elu_d4')(lay_bn)
    lay_cd4 = concatenate([lay_c3a,lay_d4a],name='concat_4')
    
    lay_d3 = Conv3DTranspose(30,(3,3,3),name='DeConv_3')(lay_cd4)
    lay_bn = BatchNormalization()(lay_d3)
    lay_d3a = ELU(name='elu_d3')(lay_bn)
    lay_cd3 = concatenate([lay_c2a,lay_d3a],name='concat_3')
    
    lay_d2 = Conv3DTranspose(20,(3,3,3),name='DeConv_2')(lay_cd3)
    lay_bn = BatchNormalization()(lay_d2)
    lay_d2a = ELU(name='elu_d2')(lay_bn)
    lay_cd2 = concatenate([lay_c1a,lay_d2a],name='concat_2')
    
    lay_d1 = Conv3DTranspose(10,(3,3,3),name='DeConv_1')(lay_cd2)
    lay_bn = BatchNormalization()(lay_d1)
    lay_d1a = ELU(name='elu_d1')(lay_bn)
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid',name='output_layer')(lay_d1a)

    return Model(lay_input,lay_out)
#%%
def BlockModel_reg(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    
    padamt = 1
    crop = Cropping2D(cropping=((0, padamt), (0, padamt)), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(10*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(10*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(10*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(10*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(10*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(10*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,4):
        lay_conv1 = Conv2D(10*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(10*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(10*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(10*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(10*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(10*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=3
    lay_deconv1 = Conv2D(10*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(10*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(10*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(10*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(10*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_stride = Conv2DTranspose(10*dd,(3,3),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
        
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(10*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(10*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(10*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(10*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(10*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_stride = Conv2DTranspose(10*dd,(3,3),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
    # classifier
    lay_out = Conv2D(1,(1,1), activation='linear',name='output_layer')(lay_act)
    
    zeropad = ZeroPadding2D(padding=((0,padamt), (0, padamt)), data_format=None)(lay_out)
    
    return Model(lay_input,zeropad)
#%%
def BlockModel_reg3D(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    
    padamt = 1
    crop = Cropping3D(cropping=((0, padamt), (0, padamt), (0,padamt)), data_format=None)(lay_input)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv3D(10*rr, (1, 1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv3D(10*rr,(1,1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv3D(10*rr,(4,4,4),padding='valid',strides=(2,2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,4):
        lay_conv1 = Conv3D(10*rr, (1, 1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv3D(10*rr,(1,1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv3D(10*rr,(4,4,4),padding='valid',strides=(2,2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=3
    lay_deconv1 = Conv3D(10*dd,(1,1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv3D(10*dd,(3,3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv3D(10*dd, (3,3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv3D(10*dd, (3,3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv3DTranspose(10*dd,(3,3,3),name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_up = UpSampling3D(size=(2, 2, 2),name='UpSample_{}'.format(dd))(lay_act)
#    lay_stride = Conv3DTranspose(10*dd,(4,4,4),strides=(2,2,2),name='DeConvStride_{}'.format(dd))(lay_act)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_up)
        
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv3D(10*dd,(1,1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv3D(10*dd,(3,3,1),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv3D(10*dd, (3, 3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv3D(10*dd, (3, 3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv3D(10*dd,(1,1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling3D(size=(2, 2, 2),name='UpSample_{}'.format(dd))(lay_act)
#        lay_stride = Conv3DTranspose(10*dd,(4,4,4),strides=(2,2,2),name='DeConvStride_{}'.format(dd))(lay_act)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_up)
    lay_touchup= Conv3DTranspose(10*4,(3,3,3),name='DeConvTouchup')(lay_act)
    lay_act = ELU(name='elu_touchup'.format(dd))(lay_touchup)
    # classifier
    lay_out = Conv3D(1,(1,1,1), activation='linear',name='output_layer')(lay_act)
    
#    zeropad = ZeroPadding3D(padding=((0,padamt), (0, padamt), (0,padamt)), data_format=None)(lay_out)
    
    return Model(lay_input,lay_out)

#%%
def BlockModel_reg3D_v2(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')

    # contracting block 1
    rr = 1
    lay_conv1 = Conv3D(10*rr, (1, 1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_input)
    lay_conv3 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_input)
    lay_conv51 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_input)
    lay_conv52 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv3D(10*rr,(1,1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv3D(10*rr,(2,2,2),padding='valid',strides=(2,2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,4):
        lay_conv1 = Conv3D(10*rr, (1, 1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv3D(10*rr, (3, 3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv3D(10*rr,(1,1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv3D(10*rr,(2,2,2),padding='valid',strides=(2,2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act) 
    
    # expanding block 3
    dd=3
    lay_deconv1 = Conv3D(10*dd,(1,1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv3D(10*dd,(3,3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv3D(10*dd, (3,3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv3D(10*dd, (3,3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv3D(10*dd,(1,1,1),name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_up = UpSampling3D(size=(2, 2, 2))(lay_act)
    lay_decon = Conv3D(10*dd,(3,3,3),padding='same',name='DeCon1_{}'.format(dd))(lay_up)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_decon)
    
    
    
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv3D(10*dd,(1,1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv3D(10*dd,(3,3,1),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv3D(10*dd, (3, 3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv3D(10*dd, (3, 3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv3D(10*dd,(1,1,1),name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling3D(size=(2, 2, 2))(lay_act)
        lay_decon = Conv3DTranspose(10*dd,(3,3,3),padding='same',name='DeCon1_{}'.format(dd))(lay_up)
        lay_decon = Conv3DTranspose(10*dd,(3,3,3),padding='same',name='DeCon2_{}'.format(dd))(lay_decon)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_decon)
        
    # regressor
    lay_out = Conv3D(1,(1,1,1), activation='linear',name='output_layer')(lay_act)
    
    return Model(lay_input,lay_out)

#%%
def BlockModel_reg3D_v3(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    # down sampling 1
    pad = ZeroPadding3D(padding=(1,1,1))(lay_input)
    down = Conv3D(8,(3,3,3),strides=(2,2,2),name='stride1',use_bias=False)(pad)
    
    # processing block 1-1
    conv = Conv3D(8,(3,3,3),padding='same',name='conv1_1',use_bias=False)(down)
    bn = BatchNormalization()(conv)
    elu = ELU()(bn)
    conv = Conv3D(8,(3,3,3),padding='same',name='conv1_2',use_bias=False)(elu)
    bn = BatchNormalization()(conv)
    act11 = ELU()(bn)
    
    # up sampling 1-1
    conv = Conv3D(8,(1,1,1))(act11)
    up = UpSampling3D(size=(2,2,2))(conv)
    upconv = Conv3D(8,(3,3,3),padding='same',use_bias=False)(up)
    elu = ELU()(upconv)
    upconv = Conv3D(1,(3,3,3),padding='same',use_bias=False)(elu)
    elu = ELU()(upconv)
    # merge 1-1
    added = Add()([lay_input, elu])
    
    # down sampling 2
    pad = ZeroPadding3D(padding=(1,1,1))(act11)
    down2 = Conv3D(16,(3,3,3),strides=(2,2,2),name='stride2',use_bias=False)(pad)
    
    # processing block 2-1
    conv = Conv3D(16,(3,3,3),padding='same',name='conv1_1',use_bias=False)(down2)
    bn = BatchNormalization()(conv)
    elu = ELU()(bn)
    conv = Conv3D(16,(3,3,3),padding='same',name='conv1_2',use_bias=False)(elu)
    bn = BatchNormalization()(conv)
    act21 = ELU()(bn)
    # up sampling 2-1
    conv = Conv3D(16,(1,1,1))(act21)
    up = UpSampling3D(size=(2,2,2))(conv)
    upconv = Conv3D(16,(3,3,3),padding='same',use_bias=False)(up)
    elu = ELU()(upconv)
    upconv = Conv3D(8,(3,3,3),padding='same',use_bias=False)(elu)
    up21 = ELU()(upconv)
    
    
    # processing block 1-2
    conv = Conv3D(8,(3,3,3),padding='same',name='conv2_1',use_bias=False)(act11)
    bn = BatchNormalization()(conv)
    elu = ELU()(bn)
    conv = Conv3D(8,(3,3,3),padding='same',name='conv2_2',use_bias=False)(elu)
    bn = BatchNormalization()(conv)
    act12 = ELU()(bn)
    
    # merge 1-2 to 21
    merge12 = concatenate([act12,up21])
    
    # up sampling 1-2
    conv = Conv3D(8,(1,1,1))(merge12)
    up = UpSampling3D(size=(2,2,2))(conv)
    upconv = Conv3D(8,(3,3,3),padding='same',use_bias=False)(up)
    elu = ELU()(upconv)
    upconv = Conv3D(1,(3,3,3),padding='same',use_bias=False)(elu)
    elu = ELU()(upconv)
    
    
       
    
    # regressor
    lay_out = Conv3D(1,(1,1,1),activation='linear',name='output')(added)
    
    return Model(lay_input,lay_out)

#%%
def TestModel(samp_input):
    # model testing
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    lay_c1 = Conv3D(10,(4,4,4),strides=(2,2,2),activation='relu')(lay_input)
    lay_merge = concatenate([lay_c1,lay_c1])
    bn = BatchNormalization()(lay_merge)
    lay_act = ELU(name='elu')(bn)
    print(lay_act.get_shape())
    lay_up = UpSampling3D(size=(2,2,2))(lay_act)
    lay_out = Conv3D(1,(1,1,1), activation='sigmoid')(lay_up)
    TestModel = Model(lay_input,lay_out)
    TestModel.summary()