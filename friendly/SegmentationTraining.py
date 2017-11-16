#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:45:31 2017

@author: jmj136
"""
import numpy as np
import skimage.exposure as skexp
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from keras.layers.convolutional import ZeroPadding2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import keras.backend as K
from matplotlib import pyplot as plt
import time
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#%% User defined variables

# Where to store the model
model_filepath = 'SegModel.hdf5'

# Data path/name
datapath = 'SegData.hdf5'
# Data must be saved in a single HDF5 file in the following format:
# two datasets: inputs "x" and targets "y"
# inputs are of shape (samples,row,col,channels)
# outputs are of shape (samples,row,col,classes)

# choose whether to augment training data or not
augment_data=False

# amount of data to use as validation data
valFrac = 0.20

# maximum number of epochs (iterations) to train
numEp = 2

# model-specifying variables

# sets the number of filters to use at each level in the model
# increase if results are poor, decrease if model is too big
filter_multiplier = 16

# sets the number of Inception modules to use in the model
# increase to get larger receptive field and deeper model
# decrease if your images are too small
numberOfBlocks = 4


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
    val_inds=np.random.choice(np.arange(numSamples),np.round(val_split*numSamples).astype(np.int),replace=False)
    # use those indices to take out validation data from training data
    val_x = np.take(x_data,val_inds,axis=0)
    val_y = np.take(y_data,val_inds,axis=0)
    # remove validation data from training data and randomize
    train_x = np.delete(x_data,val_inds,axis=0)
    train_y = np.delete(y_data,val_inds,axis=0)
    rand_inds = np.random.permutation(np.arange(x_train.shape[0]))
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
    aug_inputs = np.random.permutation(np.concatenate((inputs,fl_inputs,gm_inputs),axis=0))
    aug_targets = np.random.permutation(np.concatenate((targets,fl_targets,gm_targets),axis=0))
    return aug_inputs,aug_targets
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
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',name='CleanUp_1')(lay_pad)
    lay_cleanup = Conv2D(filt_num,(3,3),padding='same',name='CleanUp_2')(lay_cleanup)
    # classifier
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_cleanup)
    
    return Model(lay_input,lay_out)
#%% Generate and compile model
def GetModel(x_train,numFilts=16,numBlocks=3):
    Model = BlockModel(x_train.shape,numFilts,numBlocks)
    adopt = optimizers.adam()
    Model.compile(loss=dice_loss, optimizer=adopt,
                  metrics=[dice_score])
    return Model
#%% evaluate model
def EvaluateModel(Model,x_test,y_test):
    time1 = time.time()
    scores = Model.evaluate(x_test,y_test)
    time2 = time.time()
    timePerSlice = (time2-time1)/x_test.shape[0]    
    return scores,timePerSlice

#%% Viewing tool
def mask_viewer0(imvol,maskvol,name='Mask Display'):
    msksiz = np.r_[maskvol.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*maskvol.astype(float)
    
    imvol -= np.min(imvol)
    imvol /= np.max(imvol)
    
    fig = plt.figure(figsize=(5,5))
    fig.index = 0
    imobj = plt.imshow(imvol[fig.index,...],cmap='gray',aspect='equal',vmin=0, vmax=1)
    mskobj = plt.imshow(msk[fig.index,...])
    plt.tight_layout()
    plt.suptitle(name)
    ax = fig.axes[0]
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,fig.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    fig.imvol = imvol
    fig.maskvol = msk
    fig.imobj = imobj
    fig.mskobj = mskobj
    fig.txtobj = txtobj
    fig.canvas.mpl_connect('scroll_event',on_scroll_m0)
    
def on_scroll_m0(event):
    fig = event.canvas.figure
    if event.button == 'up':
        next_slice_m0(fig)
    elif event.button == 'down':
        previous_slice_m0(fig)
    fig.txtobj.set_text(fig.index+1)
    fig.canvas.draw()
    
def previous_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
#    fig.index = (fig.index - 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index-1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

def next_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
#    fig.index = (fig.index + 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index+1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()
    
#%% Main script
if __name__ == "__main__":
    print("Setting up segmentation")
    
    print("Loading data from file...")
    x_data,y_data = LoadData(datapath)
    print("Data loaded.")
    
    print("Separating data into validation and test...")
    x_train,y_train,x_val,y_val=SplitData(x_data,y_data,val_split=valFrac)
    del x_data
    del y_data
    
    if augment_data:
        print("Augmenting data...")
        x_train,y_train = generate_augmented_data(x_train,y_train)
        
    print("Setting callbacks...")
    CBs = SetCallbacks()
    
    print("Building model...")
    SegModel = GetModel(x_train,filter_multiplier,numberOfBlocks)
    print('Number of parameters: ',SegModel.count_params())
    
    print('Starting training')
    b_s = np.minimum(np.maximum(np.round(x_train.shape[0]/20),4),16).astype(np.int)
    history = SegModel.fit(x_train, y_train,
                       batch_size=b_s, epochs=numEp,
                       validation_data=(x_val,y_val),
                       verbose=1,
                       callbacks=CBs)
    
    print('Training Complete')
    
    print('Evaluating results...')
    scores,tps = EvaluateModel(SegModel,x_val,y_val)
    print('Validation Dice Score: ',scores[1])
    print('Inference time: {:.03f} ms'.format(tps*10e3))
    
    print('Plotting metrics')
    actEpochs = len(history.history['loss'])
    epochs = np.arange(1,actEpochs+1)
    
    fig2 = plt.figure(1,figsize=(12.0, 6.0));
    plt.plot(epochs,history.history['dice_score'],'r-o')
    plt.plot(epochs,history.history['val_dice_score'],'b-s')
    plt.ylim([0,1])
    plt.legend(['trainind dice score', 'validation dice score'], loc='upper left')
    plt.show()
    
    print('Displaying validation masks')
    output = SegModel.predict(x_val,batch_size=8)
    mask_viewer0(x_val[...,0],output[...,0])
    