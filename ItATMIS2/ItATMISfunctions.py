#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:52:53 2018

@author: jmj136
"""

#%% Block Segmentation Model
import numpy as np
from keras.layers import Input,Cropping2D,Conv2D,concatenate
from keras.layers import BatchNormalization,ELU,UpSampling2D
from keras.layers import Conv2DTranspose,ZeroPadding2D
from keras.models import Model

def BlockModel(in_shape,filt_num=16,numBlocks=4,num_out_channels=2):
    input_shape = in_shape[1:]
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    #calculate appropriate cropping
    mod = np.mod(input_shape[0:2],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[0:2]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        numBlocks=3
    
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
        
    # segmenter
    lay_out = Conv2D(num_out_channels,(1,1), activation='sigmoid',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)

#%% Dice Coefficient Loss
import keras.backend as K
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

#%% One Pass of iterative training simulation
from keras.preprocessing.image import ImageDataGenerator
def SimulateItATMIS(model,cur_inputs,cur_targets,CBs,val_frac=0.2):
    # split off validation data
    numIm = cur_inputs.shape[0]
    val_inds = np.random.choice(np.arange(numIm),
                                np.round(val_frac*numIm).astype(np.int),
                                replace=False)
    valX = np.take(cur_inputs,val_inds,axis=0)
    valY = np.take(cur_targets,val_inds, axis=0)
    trainX = np.delete(cur_inputs, val_inds, axis=0)
    trainY = np.delete(cur_targets, val_inds, axis=0)
    
    
    # setup image data generator
    datagen1 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    datagen2 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
     
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    datagen1.fit(trainX, seed=seed)
    datagen2.fit(trainY, seed=seed)
    batchsize = 16
    datagen = zip( datagen1.flow( trainX, None, batchsize, seed=seed), datagen2.flow( trainY, None, batchsize, seed=seed) )
    
    # calculate number of epochs and batches
    # numEp = np.maximum(40,np.minimum(np.int(10*(self.FNind+1)),100))
    numEp = 30
    steps = np.minimum(np.int(trainX.shape[0]/batchsize*16),200)
    
    model.fit_generator(datagen,
                        steps_per_epoch=steps,
                        epochs=numEp,
                        callbacks=CBs,
                        verbose=0,
                        validation_data=(valX,valY))
    return model
#%% Plot ItATMIS simulation results
from matplotlib import pyplot as plt
from glob import glob
def PlotResults(anatomy):
    txt_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV*.txt'.format(anatomy)
    result_files = glob(txt_path)
    
    scores = [np.loadtxt(f) for f in result_files]
    iters = [range(1,s.shape[0]+1) for s in scores]
    
    plt.figure()
    for it in range(len(scores)):
        plt.plot(iters[it],scores[it],'-o')
        plt.title('Dice Score over Iterations')
        plt.xlabel('Number of subjects')
        plt.ylabel('Dice')