#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:38:59 2017

@author: jmj136
"""
from keras.layers import Input, Cropping2D, Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model
import numpy as np

def BlockModel(input_shape,filt_num=16,numBlocks=3):
    # Input shape should in the form
    # [rows,cols,channels]
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    #calculate appropriate cropping
    mod = np.mod(input_shape[0:2],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[0:2]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        raise ValueError('Too small of input for this many blocks. Use fewer blocks or larger input')
    
    crop = Cropping2D(cropping=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_input)

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # rest of contracting blocks
    for rr in range(2,numBlocks+1):
        lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # last expanding block
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)    
    lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # rest of expanding blocks
    expnums = list(range(1,numBlocks))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
        
    # regressor
    lay_out = Conv2D(1,(1,1), activation='linear',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)