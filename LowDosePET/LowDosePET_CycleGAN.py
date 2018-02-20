# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from tqdm import tqdm, trange

# Use first available GPU
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

np.random.seed(seed=1)

#%%
# Model Save Path/name
model_filepath = 'LowDosePET_CyleGAN_model.h5'

# Data path/name
datapath = 'lowdosePETdata_cycleGAN.hdf5'

if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        "train_inputs"
        test_LD = np.array(f.get('test_inputs'))
        test_FD = np.array(f.get('test_targets'))
        train_LD = np.array(f.get('train_inputs'))
        train_FD = np.array(f.get('train_targets'))
        val_LD = np.array(f.get('val_inputs'))
        val_FD = np.array(f.get('val_targets'))

train_FD = np.concatenate((train_FD,val_FD))
del val_FD
        
#%% Keras imports and initializations
# Weights initializations
# bias are initailized as 0
import keras.backend as K
from keras.layers import Input, Cropping2D, Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D
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

use_bn = False

#%% Generator Model
def GeneratorModel(input_shape, output_chan):
    lay_input = Input(shape=input_shape,name='input_layer')
    
    numB=4
    noStride = 2
    padamt = 1
    lay_crop = Cropping2D(((padamt,padamt),(padamt,padamt)))(lay_input)

    filtnum = 16
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
    
    # contracting blocks 2->numB
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
        if rr > noStride:
            x = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(1,1),kernel_initializer=conv_initG,
                       name='ConvNoStride_{}'.format(rr))(x)
        else:
            x = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initG,
                       name='ConvStride_{}'.format(rr))(x)
        x = ELU(name='elu{}_stride'.format(rr))(x)
        act_list.append(x)
    
    # expanding block numB
    dd=numB
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
    if dd >noStride:
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name='elu{}d_all'.format(dd))(x)
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
    else:
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
    
    # expanding blocks (numB-1)->1
    expnums = list(range(1,numB))
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
        if dd >noStride:
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name='elu{}d_all'.format(dd))(x)
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
        else:
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
    
    # regressor    
    x = ZeroPadding2D(padding=((padamt,padamt), (padamt,padamt)), data_format=None)(x)
    lay_out = Conv2D(output_chan,(1,1), activation='linear',kernel_initializer=conv_initG,
                       name='regression')(x)
    
    return Model(lay_input,lay_out)
#%% Discriminator model
#from keras.layers import Flatten, Dense#, Activation
from keras.layers import GlobalAveragePooling2D

def DiscriminatorModel(input_shape,filtnum=16):
    # Conditional Inputs
    lay_input = Input(shape=input_shape,name='input')
    
    usebias = False
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*(2**(rr-1)), (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}'.format(rr))(lay_input)
    lay_conv3 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}'.format(rr))(lay_input)
    lay_conv51 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}'.format(rr))(lay_input)
    lay_conv52 = Conv2D(filtnum*(2**(rr-1)), (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*(2**(rr-1)),(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}'.format(rr))(lay_merge)
#    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_2'.format(rr))(lay_stride)
    
    
    # contracting blocks 2-3
    for rr in range(2,4):
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
    
    lay_one = Conv2D(1,(3,3),kernel_initializer=conv_initD,
                     use_bias=usebias,name='ConvOne')(lay_act)
    lay_avg = GlobalAveragePooling2D()(lay_one)
    
#    lay_flat = Flatten()(lay_act)
#    lay_dense = Dense(1,kernel_initializer=conv_initD,name='Dense1')(lay_flat)
    
    return Model(lay_input,lay_avg)

#%% Setup models for training
print("Generating models...")
from keras.optimizers import Adam
# set learning rates and parameters
lrD = 2e-4
lrG = 2e-4
λ = 10  # grad penalty weighting
C = 500  # Cycle Loss weighting

# Discriminator models
DisModel_FD = DiscriminatorModel(train_FD.shape[1:],32)
DisModel_LD = DiscriminatorModel(train_LD.shape[1:],32)
# Generator Models
GenModel_LD2FD = GeneratorModel(train_LD.shape[1:],train_FD.shape[-1])
GenModel_FD2LD = GeneratorModel(train_FD.shape[1:],train_LD.shape[-1])

# Endpoints of graph- LD to FD
real_LD = GenModel_LD2FD.inputs[0]
fake_FD = GenModel_LD2FD.outputs[0]
rec_LD = GenModel_FD2LD([fake_FD])
# Endpoints of graph- FD to LD
real_FD = GenModel_FD2LD.inputs[0]
fake_LD = GenModel_FD2LD.outputs[0]
rec_FD = GenModel_LD2FD([fake_LD])
# LD to FD and back generator function
fn_genFD = K.function([real_LD],[fake_FD,rec_LD])

# Discriminator scores
realFDscore = DisModel_FD([real_FD])
fakeFDscore = DisModel_FD([fake_FD])
realLDscore = DisModel_LD([real_LD])
fakeLDscore = DisModel_LD([fake_LD])

#%% FD discriminator loss function
# create mixed output for gradient penalty
ep_input1 = K.placeholder(shape=(None,1,1,1))
mixed_FD = Input(shape=train_FD.shape[1:],
                    tensor=ep_input1 * real_FD + (1-ep_input1) * fake_FD)
mixedFDscore = DisModel_FD([mixed_FD])
# discriminator losses
realFDloss = K.mean(realFDscore)
fakeFDloss = K.mean(fakeFDscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedFDscore],[mixed_FD])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_FD = fakeFDloss - realFDloss + λ * grad_penalty

#%% LD to FD generator loss function
# Adversarial Loss + LD-FD-LD cycle loss
loss_LD2FD = -fakeFDloss
loss_LD2FD2LD = K.mean(K.abs(rec_LD-real_LD))

#%% LD discriminator loss function
# create mixed output for gradient penalty
ep_input2 = K.placeholder(shape=(None,1,1,1))
mixed_LD = Input(shape=train_LD.shape[1:],
                    tensor=ep_input2 * real_LD + (1-ep_input2) * fake_LD)
mixedLDscore = DisModel_LD([mixed_LD])
# discriminator losses
realLDloss = K.mean(realLDscore)
fakeLDloss = K.mean(fakeLDscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedLDscore],[mixed_LD])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_LD = fakeLDloss - realLDloss + λ * grad_penalty

#%% FD to LD generator loss function
# Adversarial Loss + LD-FD-LD cycle loss
loss_FD2LD = -fakeLDloss
loss_FD2LD2FD = K.mean(K.abs(rec_FD-real_FD))

#%% Training functions
# Discriminator training function
loss_D = loss_FD + loss_LD
weights_D = DisModel_FD.trainable_weights + DisModel_LD.trainable_weights
D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(weights_D,[],loss_D)
fn_trainD = K.function([real_LD, real_FD, ep_input1, ep_input2],[loss_FD, loss_LD], D_trups)

# Generator Training function
loss_G = loss_LD2FD + C*loss_LD2FD2LD + loss_FD2LD + C*loss_FD2LD2FD
weights_G = GenModel_LD2FD.trainable_weights + GenModel_FD2LD.trainable_weights
LD2FD_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], loss_G)
# Generator training function returns LD2FD discriminator loss and LD2FD2LD cycle loss
fn_trainG = K.function([real_LD,real_FD], [loss_LD2FD,loss_LD2FD2LD], LD2FD_trups)

# validation evaluate function
fn_evalCycle = K.function([real_LD],[loss_LD2FD2LD])

#%% training
print('Starting training...')
ex_ind = 123
numIter = 5000
progstep = 20
valstep = 50
b_s = 8
val_b_s = 8
train_rat = 3
dis_loss = np.zeros((numIter,2))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(val_LD.shape[0]/val_b_s),2))

# Updatable plot
plt.ion()
fig, ax = plt.subplots()
LD_samp = test_LD[ex_ind,...][np.newaxis,...]
[test,rec] = fn_genFD([LD_samp])
samp_im = np.c_[LD_samp[0,...,0],test[0,...,0],test_FD[ex_ind,...,0],rec[0,...,0]]
ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
ax.set_axis_off()
ax.set_clip_box([0,1])
plt.pause(.001)
plt.draw()


print('Training adversarial model')
# preallocate image display
progress_ims = np.zeros((np.int(numIter/progstep),256,3*256))
gg = 0
vv = 0

t = trange(numIter,file=sys.stdout)
for ii in t:
    for _ in range(train_rat):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(train_LD.shape[0], b_s, replace=False)
        LD_batch = train_LD[batch_inds,...]
        batch_inds = np.random.choice(train_FD.shape[0], b_s, replace=False)
        FD_batch = train_FD[batch_inds,...]
        # train discrimator
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        ϵ2 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([LD_batch, FD_batch, ϵ1,ϵ2])
    dis_loss[ii] = errD
    # Train Generator
    errG = fn_trainG([LD_batch, FD_batch])
    gen_loss[ii] = errG
    # Update progress image
    if ii % progstep == 0:
        LD_samp = test_LD[ex_ind,...][np.newaxis,...]
        [test,rec] = fn_genFD([LD_samp])
        samp_im = np.c_[LD_samp[0,...,0],test[0,...,0],test_FD[ex_ind,...,0],rec[0,...,0]]
        progress_ims[gg] = samp_im
        ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
        plt.pause(.001)
        plt.draw()
        gg += 1
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,val_LD.shape[0]))
            LD_batch = val_LD[val_inds,...]
            valL1 = fn_evalCycle([LD_batch])[0]
            templosses[bb] = valL1
        cur_val_loss = np.mean(templosses,axis=0)
        if cur_val_loss[0] <= np.min(val_loss,axis=0)[0]:
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss[0]))
            GenModel_LD2FD.save(model_filepath,True,False)
            tqdm.write("Saved model to file")
        else:
            tqdm.write('Validation loss did not decrease: {:.02e}'.format(cur_val_loss[0]))
            
        val_loss[vv] = cur_val_loss           
        vv +=1
        
    t.set_postfix(Dloss=dis_loss[ii,0],CycleLoss = gen_loss[ii,1])
    
t.close()
del t

print('Training complete')

model_json = GenModel_LD2FD.to_json()
with open("BackupModel_LD2FD.json", "w") as json_file:
    json_file.write(model_json)
GenModel_LD2FD.save_weights("BackupModel_LD2FD.h5")

model_json = GenModel_FD2LD.to_json()
with open("BackupModel_FD2LD.json", "w") as json_file:
    json_file.write(model_json)
GenModel_FD2LD.save_weights("BackupModel_FD2LD.h5")

print('Backup Models saved')

# display loss
from scipy.signal import medfilt
fig5 = plt.figure(5)
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],5),
         np.arange(numIter),-medfilt(dis_loss[:,1],5),
         np.arange(numIter),medfilt(1000*gen_loss[:,1],5),
         np.arange(0,numIter,valstep),1000*val_loss[:,0])
plt.legend(['-Discriminator FD Loss',
            '-Discriminator LD Loss',
            '1000x Cycle Loss',
            '1000x Validation Cycle loss'])
plt.ylim([0,60])
plt.show()

#%%
print('Generating samples')
from keras.models import load_model

#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Load backup models
# from keras.models import model_from_json
#json_file = open('BackupModel_LD2FD.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#LD2FD_Model = model_from_json(loaded_model_json)
#LD2FD_Model.load_weights("BackupModel_LD2FD.h5")
# from keras.models import model_from_json
#json_file = open('BackupModel_FD2LD.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#FD2LD_Model = model_from_json(loaded_model_json)
#FD2LD_Model.load_weights("BackupModel_FD2LD.h5")
#print("Loaded backup models from weights")
#~#~#~#~#~#~#~#~#~#~#~#~#~#

GenModel = load_model(model_filepath,None,False)

# get generator results
time1 = time.time()
test_output = GenModel.predict(test_LD)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/test_LD.shape[0],' ms per slice')

# Display some samples
from VisTools import multi_slice_viewer0
if 'progress_ims' in locals():
    multi_slice_viewer0(progress_ims,'Training Progress Images')
    # save progress ims to gif
    import imageio
    output_file = 'ProgressIms.gif'
    gif_ims = np.copy(progress_ims)
    gif_ims[gif_ims<0] = 0
    gif_ims[gif_ims>1] = 1
    gif_ims = (255*gif_ims).astype(np.uint8)
    images = [gif_ims[ii,...] for ii in range(gif_ims.shape[0])]
    imageio.mimsave(output_file, images, duration=1/5,loop=1)
try:
    testFD = np.zeros((test_LD.shape[:3]))
    testRec = np.zeros_like(test_LD)
    for bb in range(0,np.int(test_LD.shape[0]/8)):
            test_inds = np.arange(bb*8,np.minimum((bb+1)*8,test_LD.shape[0]))
            LD_batch = test_LD[test_inds,...]
            [test,rec] = fn_genFD([LD_batch])
            testFD[test_inds] = test[...,0]
            testRec[test_inds] = rec
    multi_slice_viewer0(np.c_[test_LD[...,0],testFD,test_FD[...,0],testRec[...,0]],'Test Images')
except Exception as e:
    multi_slice_viewer0(np.c_[test_LD[...,0],test_output[...,0],test_FD[...,0]],'Test Images')
