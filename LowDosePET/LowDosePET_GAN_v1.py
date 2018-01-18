# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from keras import optimizers
from keras.models import load_model
import keras.backend as K
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from tqdm import trange
os.environ["CUDA_VISIBLE_DEVICES"]="1"
np.random.seed(seed=1)
#%%
# Model Save Path/name
model_filepath = 'LowDosePET_GAN_v1.hdf5'
# Data path/name
datapath = 'lowdosePETdata_v2.hdf5'
if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        x_train = np.array(f.get('train_inputs'))
        y_train = np.array(f.get('train_targets'))
        x_val = np.array(f.get('val_inputs'))
        y_val = np.array(f.get('val_targets'))
        x_test = np.array(f.get('test_inputs'))
        y_test = np.array(f.get('test_targets')) 
#%% Custom Loss
def modified_binary_crossentropy(target, output):
    #output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    #return -(target * output + (1.0 - target) * (1.0 - output))
    return K.mean(target*output)
#%% Model
from keras.layers import Input, Cropping2D, Conv2D, concatenate, add, Lambda
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import ELU
from keras.models import Model
def GeneratorModel(samp_input):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    
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
    for rr in range(2,3):
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
    dd=2
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
    expnums = list(range(1,2))
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
    
    # regressor    
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
    lay_reg = Conv2D(1,(1,1), activation='linear',name='regression')(lay_pad)
    in0 = Lambda(lambda x : x[...,0],name='channel_split')(lay_input)
    in0 = Reshape([256,256,1])(in0)
    lay_res = add([in0,lay_reg],name='residual')
    
    return Model(lay_input,lay_res)
#%% Discriminator model
from keras.layers import Flatten, Dense, Activation #,Dropout

def DiscriminatorModel(input_shape,test_shape,filtnum=16):
    # Conditional Inputs
    lay_input = Input(shape=input_shape,name='conditional_input')
    
    lay_step = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),name='StepdownLayer')(lay_input)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_step)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_step)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_step)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act1 = ELU(name='elu{}_2'.format(rr))(lay_stride)
    
    # Testing Input block
    lay_test = Input(shape=test_shape,name='test_input')
    lay_step2 = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),name='StepdownLayer2')(lay_test)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}t'.format(rr))(lay_step2)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}t'.format(rr))(lay_step2)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}t'.format(rr))(lay_step2)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}t'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}t'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}t'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1t'.format(rr))(bn)
    lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}t'.format(rr))(lay_act)
    lay_act2 = ELU(name='elu{}_2t'.format(rr))(lay_stride)
    
    # Merge blocks
    lay_act = concatenate([lay_act1,lay_act2],name='InputMerge')
    # contracting blocks 2-5
    for rr in range(2,6):
        lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    
    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1)(lay_flat)
    lay_sig = Activation('sigmoid')(lay_dense)
    
    return Model(inputs=[lay_input,lay_test],outputs=[lay_sig])

#%% prepare model for training
print("Generating models...")

adopt1 = optimizers.adam(lr=1e-5)
adopt2 = optimizers.adam(lr=1e-5)

DisModel = DiscriminatorModel(x_train.shape[1:],y_train.shape[1:],20)
DisModel.compile(optimizer=adopt1,loss=modified_binary_crossentropy)

GenModel = GeneratorModel(x_train)

DisModel.trainable = False
for l in DisModel.layers:
    l.trainable = False

GMmodel = Model(inputs=GenModel.input, outputs=DisModel([GenModel.input,GenModel.output]))

GMmodel.compile(optimizer=adopt2, loss=modified_binary_crossentropy)
#%%
# Example Display function
def display_example(ex_ind,x_train,y_train,GenModel):
    plt.figure(1,figsize=(9,3.3),frameon=False)
    plt.ion()
    cond_samp = x_train[ex_ind,...][np.newaxis,...]
    truth_samp = y_train[ex_ind]
    output_samp = GenModel.predict(cond_samp)
    samp_im = np.c_[cond_samp[0,...,0],output_samp[0,...,0],truth_samp[...,0]]
    plt.pause(0.0001)
    plt.imshow(samp_im,cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
#%% training
print('Starting training...')
ex_ind = 50
numIter = 10000
numPreIter = 101
b_s = 8
disMult = 2
softMax = .9
dis_loss = np.zeros((numIter,1))
gen_loss = np.zeros((numIter,1))

# pre train the discriminator
print('Pre-training the discriminator...')
#sys.stdout.flush

t = trange(numPreIter,file=sys.stdout)
for pp in t:
    # grab random training samples
        batch_inds = np.random.randint(0,x_train.shape[0], size=b_s)
        cond_batch = x_train[batch_inds,...]
        real_batch = y_train[batch_inds,...]
        # generate current output of generator model
        fake_batch = GenModel.predict(cond_batch,batch_size=b_s)
        # create targets for discriminator model
        y_batchR = softMax*np.ones([b_s, 1])
        y_batchF = np.zeros([b_s, 1])
        # train discrimator model on real batch
        rloss=DisModel.train_on_batch(x={'conditional_input':cond_batch,
                                   'test_input':real_batch},y=y_batchR)
        # train discrimator model on fake batch
        floss = DisModel.train_on_batch(x={'conditional_input':cond_batch,
                                    'test_input':fake_batch},y=y_batchF)
        disloss = (rloss+floss)/2
        t.set_postfix(Dloss=disloss)
        
t.close()


print('Training adversarial model')

t = trange(numIter,file=sys.stdout)
for ii in t:
    # Train Discriminator
    for _ in range(disMult):
        # grab random training samples
        batch_inds = np.random.randint(0,x_train.shape[0], size=b_s)
        cond_batch = x_train[batch_inds,...]
        real_batch = y_train[batch_inds,...]
        # generate current output of generator model
        fake_batch = GenModel.predict(cond_batch,batch_size=b_s)
        # display example image
    #    display_example(ex_ind)
        # create targets for discriminator model
        y_batchR = softMax*np.ones([b_s, 1])
        y_batchF = np.zeros([b_s, 1])
        # train discrimator model on real batch
        rloss = dis_loss[ii]=DisModel.train_on_batch(x={'conditional_input':cond_batch,
                                               'test_input':real_batch},y=y_batchR)
        # train discrimator model on fake batch
        floss = DisModel.train_on_batch(x={'conditional_input':cond_batch,
                                               'test_input':fake_batch},y=y_batchF)
        meanloss = (rloss+floss)/2
        dis_loss[ii] = meanloss
    # Train Combined Model
    batch_inds = np.random.randint(0,x_train.shape[0], size=b_s)
    cond_batch = x_train[batch_inds,...]
    y_batch = .9*np.ones([b_s, 1])
    gen_loss[ii] = GMmodel.train_on_batch(x=cond_batch,y=y_batch)
    t.set_postfix(Dloss=dis_loss[ii], Gloss=gen_loss[ii])
    
t.close()
print('Training complete')

# display example image
display_example(ex_ind,x_train,y_train,GenModel)
# display loss
fig2 = plt.figure(2)
plt.plot(np.arange(numIter),dis_loss,np.arange(numIter),gen_loss)
plt.legend(['Discriminator Loss','Generator Loss'])
plt.show()

#%%
print('Generating samples')
# regression result
pr_bs = np.minimum(16,x_test.shape[0])
time1 = time.time()
test_output = GenModel.predict(x_test,batch_size=pr_bs)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/x_test.shape[0],' ms per slice')

from skimage.measure import compare_ssim as ssim
SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_test[...,0],test_output[...,0])]

print('Mean SSIM of', np.mean(SSIMs))
print('Median SSIM of', np.median(SSIMs))
print('SSIM range of', np.round(np.min(SSIMs),3), '-', np.round(np.max(SSIMs),3))

from VisTools import multi_slice_viewer0
multi_slice_viewer0(np.c_[x_test[...,0],test_output[...,0],y_test[...,0]],SSIMs)
