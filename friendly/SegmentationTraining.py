#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:45:31 2017

@author: jmj136
"""
import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from keras.layers.convolutional import ZeroPadding2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K
from matplotlib import pyplot as plt
from CustomMetrics import dice_coef_loss
import time
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#%% User defined variables

# Where to store the model
model_filepath = 'SegModel.hdf5'

# Data path/name
datapath = 'segmentation_data.hdf5'
# Data must be saved in a single HDF5 file in the following format:
# two datasets: inputs "x" and targets "y"
# inputs are of shape (samples,x,y,channels)
# if inputs are of single channel, then (samples,x,y)
# outputs are of shape (samples,x,y,classes)
# if outputs are of single class, then (samples,x,y)

# choose whether to augment training data or not
augment_data=False

#%% Data Loading
def LoadData(datapath):
    with h5py.File(datapath,'r') as f:
        x_data = np.array(f.get('x'))
        y_data = np.array(f.get('y'))
    return x_data,y_data
#%% Data splitting into training and validation
def SplitData(x_data,y_data,val_split=.2):
    # get total number of samples
    numSamples = x_data.shape[0];
    # randomly select a portion of these samples as indices
    val_inds=np.random.choice(np.arange(numSamples),np.round(val_split*numSamples),replace=False)
    # use those indices to take out validation data from training data
    val_x = np.take(x_data,val_inds,axis=0)
    val_y = np.take(y_data,val_inds,axis=0)
    train_x = np.delete(val_x,val_inds,axis=0)
    train_y = np.delete(val_y,val_inds,axis=0)
    return train_x,train_y,val_x,val_y
#%% Callbacks
def SetCallbacks():
    earlyStopping = EarlyStopping(monitor='val_loss',patience=10, verbose=1,mode='auto')

    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)    
    CBs = [checkpoint,earlyStopping]
    return CBs
#%% Custom Loss function based on Dice Coefficient
def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
#%% Model Architecture
def BlockModel(samp_input):
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
#%% Generate and compile model
def GetModel(x_train):
    Model = BlockModel(x_train)
    adopt = optimizers.adam()
    Model.compile(loss=dice_coef_loss, optimizer=adopt,
                  metrics=[dice_loss])
    return Model
#%% evaluate model
def EvaluateModel(Model,x_test,y_test):
    time1 = time.time()
    scores = Model.evaluate(x_test,y_test)
    time2 = time.time()
    timePerSlice = (time2-time1)/x_test.shape[0]    
    return scores,timePerSlice

if __name__ == "__main__":
    print("Setting up segmentation")
    
    print("Loading data from file...")
    x_data,y_data = LoadData(datapath)
    print("Data loaded.")
    
    print("Separating data into validation and test...")
    x_train,y_train,x_val,y_val=SplitData(x_data,y_data)
    
    if augment_data:
        print("Augmenting data...")
        
        print("not currently available")
        
    print("Setting callbacks...")
    CBs = SetCallbacks()
    
    print("Building model...")
    SegModel = GetModel(x_train)
    
    print('Starting training')
    numEp = 20
    b_s = np.minimum(np.maximum(np.round(x_train.shape[0]/20),4),16)
    history = SegModel.fit(x_train, y_train,
                       batch_size=b_s, epochs=numEp,
                       validation_data=(x_val,y_val),
                       verbose=1,
                       callbacks=CBs)
    
    print('Training Complete')
    
    print('Plotting results...')
    
    print('Plotting metrics')
    actEpochs = len(history.history['loss'])
    epochs = np.arange(1,actEpochs+1)
    
    fig2 = plt.figure(1,figsize=(12.0, 6.0));
    plt.plot(epochs,history.history['loss'],'r-o')
    plt.plot(epochs,history.history['val_loss'],'b-s')
    plt.legend(['dice loss', 'dice validation loss'], loc='upper left')
    plt.show()