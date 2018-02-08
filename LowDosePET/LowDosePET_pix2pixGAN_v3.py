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
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

np.random.seed(seed=1)

#%%
# Model Save Path/name
model_filepath = 'LowDosePET_pix2pixModel_30s.h5'
#model_filepath = 'LowDosePET_pix2pixModel_60s.h5'

# Data path/name
datapath = 'lowdosePETdata_30s.hdf5'
#datapath = 'lowdosePETdata_60s.hdf5'

MS = 3
MSoS = 1

if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        x_test = np.array(f.get('test_inputs'))
        y_test = np.array(f.get('test_targets'))
        x_train = np.array(f.get('train_inputs'))
        y_train = np.array(f.get('train_targets'))
        x_val = np.array(f.get('val_inputs'))
        y_val = np.array(f.get('val_targets'))
        
#%% Keras imports and initializations
# Weights initializations
# bias are initailized as 0
import keras.backend as K
from keras.layers import Input, Cropping2D, Conv2D, concatenate, add, Lambda
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Conv3D, Reshape
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
def GeneratorModel(input_shape):
    lay_input = Input(shape=input_shape,name='input_layer')
    
    padamt = 1
    MSconv = Conv3D(16,(3,3,3),padding='valid',name='MSconv')(lay_input)
    if use_bn:
        bn = batchnorm()(MSconv, training=1)
        MSact = ELU(name='MSelu')(bn)
    else:
        MSact = ELU(name='MSelu')(MSconv)
    MSconvRS = Reshape((254,254,16))(MSact)

    filtnum = 20
    # contracting block 1
    rr = 1
    x1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initG,
                       name='Conv1_{}'.format(rr))(MSconvRS)
    x1 = ELU(name='elu{}_1'.format(rr))(x1)
    x3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv3_{}'.format(rr))(MSconvRS)
    x3 = ELU(name='elu{}_3'.format(rr))(x3)
    x51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv51_{}'.format(rr))(MSconvRS)
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
    
    # contracting blocks 2-3
    for rr in range(2,4):
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
    x = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(x)
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
    
    MSconv = Conv3D(8,(3,3,3),padding='valid',name='MSconv')(lay_cond_input)
    MSact = LeakyReLU(name='MSleaky')(MSconv)
    MSconvRS = Reshape((254,254,8))(MSact)    
    xcond = Conv2D(filtnum,(3,3),padding='same',strides=(1,1),kernel_initializer=conv_initD,
                       name='FirstCondLayer')(MSconvRS)
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

# weighted L1 loss tensors and masks
y_true = K.flatten(real_B)
y_pred = K.flatten(fake_B)

tis_mask1 = K.cast( K.greater( y_true, 0.01 ), 'float32' )
tis_mask2 = K.cast( K.less( y_true, 0.3 ), 'float32' )
tis_mask = tis_mask1 * tis_mask2
les_mask1 =  K.cast( K.greater(y_true,0.3), 'float32' )
les_mask2 = K.cast( K.less(y_true,0.5), 'float32' )
les_mask = les_mask1 * les_mask2
les_maskB = K.cast (K.greater(y_true,.5), 'float32' )
air_mask =  K.cast( K.less( y_true, 0.01 ), 'float32' )

tis_true = tis_mask * y_true
tis_pred = tis_mask * y_pred

air_true = air_mask * y_true
air_pred = air_mask * y_pred

les_true = les_mask * y_true
les_pred = les_mask * y_pred

lesB_true = les_maskB * y_true
lesB_pred = les_maskB * y_pred

tis_loss = K.mean(K.abs(tis_true - tis_pred), axis=-1)
air_loss = K.mean(K.abs(air_true - air_pred), axis=-1)
les_loss = K.mean(K.abs(les_true - les_pred), axis=-1)
lesB_loss = K.mean(K.abs(lesB_true - lesB_pred), axis=-1)
# weighted L1 loss
loss_L1w = .05*air_loss + .15*tis_loss + .6 * les_loss + .2 * lesB_loss

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_G = -loss_D_fake + 30 * loss_L1w
training_updates = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(GenModel.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G, loss_L1], training_updates)

netG_eval = K.function([real_A, real_B],[loss_L1])

#%% training
print('Starting training...')
ex_ind = 50
numIter = 2000
progstep = 50
valstep = 50
b_s = 8
val_b_s = 8
train_rat = 5
dis_loss = np.zeros((numIter))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(x_val.shape[0]/val_b_s),2))

# Updatable plot
plt.ion()
fig, ax = plt.subplots()
cond_samp = x_test[ex_ind,...][np.newaxis,...]
simfulldose_im = GenModel.predict(cond_samp)[0,...,0]
fulldose_im = y_test[ex_ind,...,0]
samp_im = np.c_[cond_samp[0,1,...,0],simfulldose_im,fulldose_im]
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
        batch_inds = np.random.choice(x_train.shape[0], b_s, replace=False)
        cond_batch = x_train[batch_inds,...]
        real_batch = y_train[batch_inds,...]
        # train discrimator
        ϵ = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = netD_train([cond_batch, real_batch, ϵ])
    dis_loss[ii] = errD[0]
    # Train Generator
    errG = netG_train([cond_batch, real_batch])
    gen_loss[ii] = errG
    if ii % progstep == 0:
        cond_samp = x_test[ex_ind,...][np.newaxis,...]
        simfulldose_im = GenModel.predict(cond_samp)[0,...,0]
        fulldose_im = y_test[ex_ind,...,0]
        samp_im = np.c_[cond_samp[0,1,...,0],simfulldose_im,fulldose_im]
        progress_ims[gg] = samp_im
        ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
        plt.pause(.001)
        plt.draw()
        gg += 1
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
            
        val_loss[vv] = cur_val_loss           
        vv +=1
        
    t.set_postfix(Dloss=dis_loss[ii], Gloss=gen_loss[ii,0], L1loss = gen_loss[ii,1])
    
t.close()
del t

print('Training complete')

model_json = GenModel.to_json()
with open("BackupModel.json", "w") as json_file:
    json_file.write(model_json)
GenModel.save_weights("BackupModel.h5")
print('Backup Model saved')

# display loss
from scipy.signal import medfilt
fig5 = plt.figure(5)
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],5),
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
if 'progress_ims' in locals():
    multi_slice_viewer0(progress_ims,'Training Progress Images')
multi_slice_viewer0(np.c_[x_test[:,1,...,0],test_output[...,0],y_test[...,0]],'Test Images',SSIMs)
