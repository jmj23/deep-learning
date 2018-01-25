# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import load_model
from keras.metrics import mean_absolute_error 
from my_callbacks import Histories
import numpy as np
import h5py
import time
from CustomMetrics import weighted_mae
os.environ["CUDA_VISIBLE_DEVICES"]="1" # Pick GPU

numEp = 40  # Set maximum number of Epochs
b_s = 4 # Set batch size

#%%
# Model Save Path/name- where it is stored during training
model_filepath = 'LowDosePETModel_30s_v3.hdf5'
# Data path/name- which data to use
datapath = 'lowdosePETdata_v3.hdf5'

print('Loading data...')
with h5py.File(datapath,'r') as f:
    x_test = np.array(f.get('test_inputs'))
    y_test = np.array(f.get('test_targets'))
    x_val = np.array(f.get('val_inputs'))
    y_val = np.array(f.get('val_targets'))
    x_train = np.array(f.get('train_inputs'))
    y_train = np.array(f.get('train_targets'))
    
#%% Model definition
from keras.layers import Input, Cropping2D, Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.layers import add, Lambda, Reshape

def ResidualBlockModel(input_shape):
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    padamt = 1
    crop = Cropping2D(cropping=((0, padamt), (0, padamt)), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(16*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(16*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(16*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(16*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(16*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(16*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,4):
        lay_conv1 = Conv2D(16*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(16*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(16*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(16*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(16*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(16*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=3
    lay_deconv1 = Conv2D(16*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(16*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(16*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(16*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(16*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)
    
    lay_cleanup = Conv2DTranspose(16*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(16*dd, (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(16*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(16*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(16*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(16*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(16*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(16*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(16*dd, (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
        
    # regressor
    lay_reg = Conv2D(1,(1,1), activation='linear',name='reg_output')(lay_pad)
    in0 = Lambda(lambda x : x[...,0],name='channel_split')(lay_input)
    in0 = Reshape([256,256,1])(in0)
    lay_res = add([in0,lay_reg],name='residual')
    
    return Model(lay_input,lay_res)
    
#%% callbacks
print('Setting callbacks...')
earlyStopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

hist = Histories()

CBs = [checkpoint,earlyStopping,hist]

#%% prepare model for training
print("Generating model...")

RegModel = ResidualBlockModel(x_train.shape[1:])
adopt = optimizers.adam()
RegModel.compile(optimizer=adopt,loss= weighted_mae, metrics= [mean_absolute_error])

#%% training
print('Starting training...')

history = RegModel.fit(x_train,y_train,
                       batch_size=b_s, epochs=numEp,
                       validation_data=(x_val,y_val),
                       verbose=1,
                       callbacks=CBs)

print('Training complete')

print('Loading best model...')
RegModel = load_model(model_filepath,custom_objects={'weighted_mae':weighted_mae})

score = RegModel.evaluate(x_test,y_test)
print("")
print("Metrics on test data: {}".format(score))

#%%
print('Generating samples')

# regression result
pr_bs = np.minimum(16,x_test.shape[0])
time1 = time.time()
test_output = RegModel.predict(x_test,batch_size=pr_bs)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/x_test.shape[0],' ms per slice')

from skimage.measure import compare_ssim as ssim
SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_test[...,0],test_output[...,0])]

print('Mean SSIM of', np.mean(SSIMs))
print('Median SSIM of', np.median(SSIMs))
print('SSIM range of', np.round(np.min(SSIMs),3), '-', np.round(np.max(SSIMs),3))
print('Standard Deviation of',np.std(SSIMs))


from VisTools import multi_slice_viewer0
multi_slice_viewer0(np.c_[x_test[...,0],x_test[...,1],test_output[...,0],y_test[...,0]],SSIMs)

#Export to NIFTI
#import nibabel as nib
#testsubj1 = np.rollaxis(np.rollaxis(test_output[:85,...,0],2,0),2,0)
#output_img = nib.Nifti1Image(testsubj1, np.eye(4))
#output_img.to_filename('subj014_simFullDosePET_30s.nii')
