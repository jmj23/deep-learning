# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12

@author: Jacob Johnson
"""
import sys
import os
# path to VisTools.py must be added or final test visualizations
# will not work
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
# for progress bar
from tqdm import tqdm, trange

# Use first available GPU
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# seeding for batches
np.random.seed(seed=1)

# Model Save Path/name
model_filepath = 'LowDosePET_pix2pixModel_30s.h5'
#model_filepath = 'LowDosePET_pix2pixModel_60s.h5'

# Data path/name
datapath = 'lowdosePETdata_30s.hdf5'
#datapath = 'lowdosePETdata_60s.hdf5'

# load data if not already loaded
if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        x_test = np.array(f.get('test_inputs'))
        y_test = np.array(f.get('test_targets'))
        x_train = np.array(f.get('train_inputs'))
        y_train = np.array(f.get('train_targets'))
        x_val = np.array(f.get('val_inputs'))
        y_val = np.array(f.get('val_targets'))
        
numIter = 1000  # number of batches to train
L1Weight = 100 # how much to weight L1 loss vs adversarial loss        
#%% Keras imports and initializations
# Weights initializations
# bias are initailized as 0
import keras.backend as K
from keras.layers import Input, Cropping2D, Conv2D, concatenate, add, Lambda
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Model
from keras.initializers import RandomNormal

#conv_initG = RandomNormal(0, 0.02)
conv_initG = 'glorot_uniform'
#conv_initD = RandomNormal(0, 0.02)
conv_initD = 'he_normal'

gamma_init = RandomNormal(1., 0.02) # for batch normalization

def batchnorm():
    return BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
# currently no BN is working better than with BN
use_bn = False

#%% Generator Model
def GeneratorModel(input_shape):
    lay_input = Input(shape=input_shape,name='input_layer')
    
    # adjust this if using different image size than 256x256
    padamt = 1
    lay_crop = Cropping2D(padding=((padamt,padamt), (padamt,padamt)), data_format=None)(lay_input)
    
    # filter multiplier to use
    filtnum = 20
    # number of "blocks" to use
    numB = 3
    
    # contracting block 1
    rr = 1
    x1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initG,
                       name='Conv1_{}'.format(rr))(lay_crop)
    x1 = ELU(name='elu{}_1'.format(rr))(x1)
    x3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv3_{}'.format(rr))(lay_crop)
    x3 = ELU(name='elu{}_3'.format(rr))(x3)
    x51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv51_{}'.format(rr))(lay_crop)
    x51 = ELU(name='elu{}_51'.format(rr))(x51)
    x52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv52_{}'.format(rr))(x51)
    x52 = ELU(name='elu{}_52'.format(rr))(x52)
    lay_merge = concatenate([x1,x3,x52],name='merge_{}'.format(rr))
    x = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='ConvAll_{}'.format(rr))(lay_merge)
    if use_bn:
        x = batchnorm()(x, training=1)
    x = ELU(name='elu{}_all'.format(rr))(x)
    x = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initG,
                       name='ConvStride_{}'.format(rr))(x)
    x = ELU(name='elu{}_stride'.format(rr))(x)
    act_list = [x]
    
    # contracting blocks 2-numB
    for rr in range(2,numB+1):
        x1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initG,
                       name='Conv1_{}'.format(rr))(x)
        x1 = ELU(name='elu{}_1'.format(rr))(x1)
        x3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv3_{}'.format(rr))(x)
        x3 = ELU(name='elu{}_3'.format(rr))(x3)
        x51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv51_{}'.format(rr))(x)
        x51 = ELU(name='elu{}_51'.format(rr))(x51)
        x52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv52_{}'.format(rr))(x51)
        x52 = ELU(name='elu{}_52'.format(rr))(x52)
        x = concatenate([x1,x3,x52],name='merge_{}'.format(rr))
        x = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='ConvAll_{}'.format(rr))(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name='elu{}_all'.format(rr))(x)
        x = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initG,
                       name='ConvStride_{}'.format(rr))(x)
        x = ELU(name='elu{}_stride'.format(rr))(x)
        act_list.append(x)
    
    # expanding block 3
    dd=3
    x1 = Conv2D(filtnum*dd,(1,1),padding='same',kernel_initializer=conv_initG,
                       name='DeConv1_{}'.format(dd))(x)
    x1 = ELU(name='elu{}d_1'.format(dd))(x1)
    x3 = Conv2D(filtnum*dd,(3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv3_{}'.format(dd))(x)
    x3 = ELU(name='elu{}d_3'.format(dd))(x3)
    x51 = Conv2D(filtnum*dd, (3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv51_{}'.format(dd))(x)
    x51 = ELU(name='elu{}d_51'.format(dd))(x51)
    x52 = Conv2D(filtnum*dd, (3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv52_{}'.format(dd))(x51)
    x52 = ELU(name='elu{}d_52'.format(dd))(x52)
    x = concatenate([x1,x3,x52],name='merge_d{}'.format(dd))
    x = Conv2D(filtnum*dd,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='DeConvAll_{}'.format(dd))(x)
    if use_bn:
        x = batchnorm()(x, training=1)
    x = ELU(name='elu{}d_all'.format(dd))(x)    
    x = UpSampling2D()(x)
    x = Conv2DTranspose(filtnum*dd, (3, 3),kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_1'.format(dd))(x)
    if use_bn:
        x = batchnorm()(x, training=1)
    x = ELU(name='elu_cleanup{}_1'.format(dd))(x)
    x = Conv2D(filtnum*dd, (3,3), padding='same',kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_2'.format(dd))(x)
    if use_bn:
        x = batchnorm()(x, training=1)
    x = ELU(name='elu_cleanup{}_2'.format(dd))(x)
    
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        x = concatenate([act_list[dd-1],x],name='skip_connect_{}'.format(dd))
        x1 = Conv2D(filtnum*dd,(1,1),padding='same',kernel_initializer=conv_initG,
                       name='DeConv1_{}'.format(dd))(x)
        x1 = ELU(name='elu{}d_1'.format(dd))(x1)
        x3 = Conv2D(filtnum*dd,(3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv3_{}'.format(dd))(x)
        x3 = ELU(name='elu{}d_3'.format(dd))(x3)
        x51 = Conv2D(filtnum*dd, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv51_{}'.format(dd))(x)
        x51 = ELU(name='elu{}d_51'.format(dd))(x51)
        x52 = Conv2D(filtnum*dd, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv52_{}'.format(dd))(x51)
        x52 = ELU(name='elu{}d_52'.format(dd))(x52)
        x = concatenate([x1,x3,x52],name='merge_d{}'.format(dd))
        x = Conv2D(filtnum*dd,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='DeConvAll_{}'.format(dd))(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name='elu{}d_all'.format(dd))(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(filtnum*dd, (3, 3),kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_1'.format(dd))(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name='elu_cleanup{}_1'.format(dd))(x)
        x = Conv2D(filtnum*dd, (3,3), padding='same',kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_2'.format(dd))(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name='elu_cleanup{}_2'.format(dd))(x)
    
    # regressor    
    x = ZeroPadding2D(padding=((padamt,padamt), (padamt,padamt)), data_format=None)(x)
    x = Conv2D(1,(1,1), activation='linear',kernel_initializer=conv_initG,
                       name='regression')(x)
    in0 = Lambda(lambda x : x[:,1,...,0],name='channel_split')(lay_input)
    in0 = Reshape([256,256,1])(x)
    lay_res = add([in0,x],name='residual')
    
    return Model(lay_input,lay_res)
#%% Discriminator model
from keras.layers import Flatten, Dense#, Activation

def DiscriminatorModel(input_shape,test_shape,filtnum=16):
    # Conditional Inputs
    lay_cond_input = Input(shape=input_shape,name='conditional_input') 
    xcond = Conv2D(filtnum,(3,3),padding='same',strides=(1,1),kernel_initializer=conv_initD,
                       name='FirstCondLayer')(lay_cond_input)
    xcond = LeakyReLU(alpha=0.2,name='leaky_cond')(xcond)
    
    
    usebias = False
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}'.format(rr))(xcond)
    lay_conv3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}'.format(rr))(xcond)
    lay_conv51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}'.format(rr))(xcond)
    lay_conv52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}'.format(rr))(lay_merge)
#    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}'.format(rr))(lay_act)
    lay_act1 = LeakyReLU(alpha=0.2,name='leaky{}_2'.format(rr))(lay_stride)
    
    
    # Testing Input block
    lay_test_input = Input(shape=test_shape,name='test_input')
    xtest = Conv2D(filtnum,(3,3),padding='valid',strides=(1,1),kernel_initializer=conv_initD,
                       use_bias=usebias,name='FirstTestLayer')(lay_test_input)
    xtest = LeakyReLU(alpha=0.2,name='leaky_test')(xtest)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}t'.format(rr))(xtest)
    lay_conv3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}t'.format(rr))(xtest)
    lay_conv51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}t'.format(rr))(xtest)
    lay_conv52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}t'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}t'.format(rr))
    lay_conv_all = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}t'.format(rr))(lay_merge)
#    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1t'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}t'.format(rr))(lay_act)
    lay_act2 = LeakyReLU(alpha=0.2,name='leaky{}_2t'.format(rr))(lay_stride)
    
    # Merge blocks
    lay_act = concatenate([lay_act1,lay_act2],name='InputMerge')
    # contracting blocks 2-5
    for rr in range(2,6):
        lay_conv1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}'.format(rr))(lay_merge)
#        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1'.format(rr))(lay_conv_all)
        lay_stride = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = LeakyReLU(alpha=0.2,name='leaky{}_2'.format(rr))(lay_stride)
    
    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1,kernel_initializer=conv_initD,name='Dense1')(lay_flat)
    
    return Model(inputs=[lay_cond_input,lay_test_input],outputs=[lay_dense])

#%% prepare model for training
print("Generating models...")
from keras.optimizers import Adam
# set learning rates and parameters
lrD = 2e-4
lrG = 2e-4
λ = 10  # grad penalty weighting

# create models
# takes filter multiplier as 3rd argument
DisModel = DiscriminatorModel(x_train.shape[1:],y_train.shape[1:],8)

GenModel = GeneratorModel(x_train.shape[1:])

# get tensors of inputs and outputs
real_A = GenModel.input
fake_B = GenModel.output
real_B = DisModel.inputs[1]
output_D_real = DisModel([real_A, real_B])
output_D_fake = DisModel([real_A, fake_B])
# create mixed output for gradient penalty
ep_input = K.placeholder(shape=(None,1,1,1))
mixed_B = Input(shape=y_train.shape[1:],
                    tensor=ep_input * real_B + (1-ep_input) * fake_B)
output_D_mixed = DisModel([real_A,mixed_B])
# discriminator losses
loss_D_real = K.mean(output_D_real)
loss_D_fake = K.mean(output_D_fake)
# gradient penalty loss
grad_mixed = K.gradients([output_D_mixed],[mixed_B])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss, training updates, and training function
loss_D = loss_D_fake - loss_D_real + λ * grad_penalty
training_updates = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(DisModel.trainable_weights,[],loss_D)
netD_train = K.function([real_A, real_B, ep_input],[loss_D], training_updates)
evalDloss = K.function([real_A,real_B,ep_input],[loss_D])

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_G = -loss_D_fake + L1Weight * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(GenModel.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G, loss_L1], training_updates)

netG_eval = K.function([real_A, real_B],[loss_L1])

#%% training
print('Starting training...')

valstep = 100   # how many iterations between validation checks
b_s = 8         # training batch size
val_b_s = 8     # validation batch size
train_rat = 5   # ratio between D and G training. Smaller is faster but
                # potentially lower quality
                
# preallocation of loss vectors for plotting
dis_loss = np.zeros((numIter))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(x_val.shape[0]/val_b_s),2))

print('Training adversarial model')
# counter for validation
vv = 0
# master counter, used in progress bar
t = trange(numIter,file=sys.stdout)
for ii in t:
    for _ in range(train_rat):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(x_train.shape[0], b_s, replace=False)
        cond_batch = x_train[batch_inds,...]
        real_batch = y_train[batch_inds,...]
        # train discrimator
        ϵ = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = netD_train([cond_batch, real_batch, ϵ])
    # record discriminator loss
    dis_loss[ii] = errD[0]
    # Train Generator
    errG = netG_train([cond_batch, real_batch])
    # record generator loss
    gen_loss[ii] = errG
    # validation checks
    # currently set to use L1 loss as validation
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,x_val.shape[0]))
            cond_batch = x_val[val_inds,...]
            real_batch = y_val[val_inds,...]
            ϵ = np.random.uniform(size=(b_s, 1, 1 ,1))
            valL1 = netG_eval([cond_batch,real_batch])[0]
            valDloss = evalDloss([cond_batch,real_batch,ϵ])[0]
            templosses[bb] = [valL1,valDloss]
        cur_val_loss = np.mean(templosses,axis=0)
        if cur_val_loss[0] <= np.min(val_loss,axis=0)[0]:
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss[0]))
            GenModel.save(model_filepath,True,False)
            tqdm.write("Saved model to file")
        else:
            tqdm.write('Validation loss did not decrease: {:.02e}'.format(cur_val_loss[0]))
            
        val_loss[vv] = cur_val_loss           
        vv +=1
    # updating progress bar
    t.set_postfix(Dloss=dis_loss[ii], Gloss=gen_loss[ii,0], L1loss = gen_loss[ii,1])
# destroy progress bar after done training
t.close()
del t

print('Training complete')
# save a backup copy of model, just in case
# validation checkpointing has issues loading
model_json = GenModel.to_json()
with open("BackupModel.json", "w") as json_file:
    json_file.write(model_json)
GenModel.save_weights("BackupModel.h5")
print('Backup Model saved')

# display loss plots
from scipy.signal import medfilt
fig5 = plt.figure(5)
plt.plot(np.arange(numIter),-medfilt(dis_loss,5),
         np.arange(numIter),medfilt(gen_loss[:,0],5),
         np.arange(numIter),medfilt(100*gen_loss[:,1],5),
         np.arange(0,numIter,valstep),100*val_loss[:,0])
plt.legend(['-Discriminator Loss',
            'Generator Loss',
            '100x L1 loss',
            '100x Validation L1 loss'])
plt.ylim([0,100])
plt.show()

#%%
print('Generating samples')
from keras.models import load_model

#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Load backup model
# from keras.models import model_from_json
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#TestModel = model_from_json(loaded_model_json)
#TestModel.load_weights("model.h5")
#print("Loaded model from disk")
#~#~#~#~#~#~#~#~#~#~#~#~#~#

GenModel = load_model(model_filepath,None,False)

# get generator results
time1 = time.time()
test_output = GenModel.predict(x_test)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/x_test.shape[0],' ms per slice')
# calculate SSIMs
from skimage.measure import compare_ssim as ssim
SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_test[...,0],test_output[...,0])]

print('Mean SSIM of', np.mean(SSIMs))
print('Median SSIM of', np.median(SSIMs))
print('SSIM range of', np.round(np.min(SSIMs),3), '-', np.round(np.max(SSIMs),3))

# Display some samples
from VisTools import multi_slice_viewer0
multi_slice_viewer0(np.c_[x_test[:,1,...,0],test_output[...,0],y_test[...,0]],'Test Images',SSIMs)
