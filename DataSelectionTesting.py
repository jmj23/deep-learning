#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:45:31 2017

@author: jmj136
"""
import numpy as np
np.random.seed(seed=1)
import skimage.exposure as skexp
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from keras.layers.convolutional import ZeroPadding2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K
from matplotlib import pyplot as plt
from CustomMetrics import dice_coef_loss
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#%% User defined variables

# Where to store the model
model_filepath = 'SegModel.hdf5'

# Data path/name
datapath = 'SelectionData.hdf5'

# amount of data to use as validation data
valFrac = 0.20

# maximum number of epochs (iterations) to train
numEp = 30

# model-specifying variables

# sets the number of filters to use at each level in the model
# increase if results are poor, decrease if model is too big
filter_multiplier = 10

# sets the number of Inception modules to use in the model
# increase to get larger receptive field and deeper model
# decrease if your images are too small
numberOfBlocks = 3


#%% Data Loading
def LoadData(datapath):
    with h5py.File(datapath,'r') as f:
        x_data = np.array(f.get('x_data'))
        y_data = np.array(f.get('y_data'))
        x_test = np.array(f.get('x_test'))
        y_test = np.array(f.get('y_test'))
    return x_data,y_data,x_test,y_test
#%% Data splitting into training and validation
def SplitData(x_data,y_data,val_split=.2,amount_of_data=1):
    # get total number of samples
    numSamples = np.round(amount_of_data*x_data.shape[0]).astype(np.int);
    x_data = x_data[:numSamples,...]
    y_data = y_data[:numSamples,...]    
    
    # randomly select a portion of these samples as indices
    val_inds=np.random.choice(np.arange(numSamples),np.round(val_split*numSamples).astype(np.int),replace=False)
    # use those indices to take out validation data from training data
    val_x = np.take(x_data,val_inds,axis=0)
    val_y = np.take(y_data,val_inds,axis=0)
    # remove validation data from training data and randomize
    train_x = np.delete(x_data,val_inds,axis=0)
    train_y = np.delete(y_data,val_inds,axis=0)
    rand_inds = np.random.permutation(np.arange(train_x.shape[0]))
    train_x = np.take(train_x,rand_inds,axis=0)
    train_y = np.take(train_y,rand_inds,axis=0)
    return train_x,train_y,val_x,val_y
#%% data augmentation, if desired
def generate_augmented_data(inputs,targets):
    # LR flips
    fl_inputs = np.flip(inputs,2)
    fl_targets = np.flip(targets,2)
    
    # gamma corrections
    gammas = .5 + np.random.rand(inputs.shape[0])
    gm_inputs = np.copy(inputs)
    for ii in range(gm_inputs.shape[0]):
        gm_inputs[ii,...,0] = skexp.adjust_gamma(gm_inputs[ii,...,0],gamma=gammas[ii])
        gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    gm_targets = np.copy(targets)
    rand_inds = np.random.permutation(np.arange(3*inputs.shape[0]))
    aug_inputs = np.take(np.concatenate((inputs,fl_inputs,gm_inputs),axis=0),rand_inds,axis=0)
    aug_targets = np.take(np.concatenate((targets,fl_targets,gm_targets),axis=0),rand_inds,axis=0)
    return aug_inputs,aug_targets
#%% Callbacks
def SetCallbacks():
    earlyStopping = EarlyStopping(monitor='val_loss',patience=6, verbose=1,mode='auto')

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
def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
#%% Model Architecture
def BlockModel(input_shape,filt_num=16,numBlocks=3):
    lay_input = Input(shape=(input_shape[1:]),name='input_layer')
        
     #calculate appropriate cropping
    mod = np.mod(input_shape[1:3],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[1:3]-padamt)
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
    
    # contracting blocks 2-n 
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
        
    # expanding block n
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
        
    # expanding blocks n-1
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
        lay_stride = Conv2DTranspose(filt_num*dd,(4,4),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
        lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
                
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)
#%% Generate and compile model
def GetModel(x_train,numFilts=16,numBlocks=3):
    Model = BlockModel(x_train.shape,numFilts,numBlocks)
    adopt = optimizers.adam()
    Model.compile(loss=dice_coef_loss, optimizer=adopt,
                  metrics=[dice_score])
    return Model
#%%
if __name__ == "__main__":
    print("Setting up segmentation")
    
    print("Loading data from file...")
    x_data,y_data,x_test,y_test = LoadData(datapath)
    print("Data loaded.")
    
    #Setup loop
    # Make array of data amounts
    a = np.linspace(.1,1,10)
    amts = np.empty((2*a.size), dtype=a.dtype)
    amts[0::2] = a
    amts[1::2] = a
    numIter = amts.size
    # make array of whether to include augmentation
    b = np.zeros((a.size), dtype=bool)
    c = np.ones((a.size), dtype=bool)
    augs = np.empty((numIter), dtype=b.dtype)
    augs[0::2] = b
    augs[1::2] = c
    results = np.zeros((numIter,3))
    
    ii = 0;
    for _ in range(2):
        for it in range(numIter):
            print("Splitting data")
            x_train,y_train,x_val,y_val=SplitData(x_data,y_data,val_split=valFrac,amount_of_data=amts[it])
            
            if augs[it]:
                print("Augmenting data...")
                x_train,y_train = generate_augmented_data(x_train,y_train)
                
            print("Setting callbacks...")
            CBs = SetCallbacks()
            
            print("Building model...")
            SegModel = GetModel(x_train,filter_multiplier,numberOfBlocks)
            
            print('Starting training')
            b_s = 16
            history = SegModel.fit(x_train, y_train,
                               batch_size=b_s, epochs=numEp,
                               validation_data=(x_val,y_val),
                               verbose=1,
                               callbacks=CBs)
            
            print('Training Complete')
            print('Loading best model...')
            SegModel = load_model(model_filepath,
                              custom_objects={'dice_score':dice_score,
                                              'dice_coef_loss':dice_coef_loss})
            print('Evalulating...')
            scores = SegModel.evaluate(x_test,y_test)
            results[ii,:] = np.array([amts[ii],augs[ii],scores[1]])
            ii +=1
            
    fig1 = plt.figure(1,figsize=(12.0, 6.0));
    plt.plot(results[0::2,0],results[0::2,2],'ro')
    plt.plot(results[1::2,0],results[1::2,2],'bo')
    plt.ylim([0,1])
    plt.title('Dice Score vs Fraction of Data used')
    plt.legend(['No data augmentation','Data Augmentation'], loc='upper left')
    plt.show()
    
    