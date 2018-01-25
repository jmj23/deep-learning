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
os.environ["CUDA_VISIBLE_DEVICES"]="1"
np.random.seed(seed=1)
#%%
# Model Save Path/name
model_filepath = 'DeepMuMap_pix2pixModel.hdf5'
# Data path/name
datapath = 'petrecondata_tvt_v3.hdf5'

if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        x_train = np.array(f.get('train_inputs'))
        y_train = np.array(f.get('train_reg_targets'))
#        y_class_train = np.array(f.get('train_class_targets'))
        x_val = np.array(f.get('val_inputs'))
        y_val = np.array(f.get('val_reg_targets'))
#        y_class_val = np.array(f.get('val_class_targets'))
        x_test = np.array(f.get('test_inputs'))
        y_test = np.array(f.get('test_reg_targets'))    
#        y_class_test = np.array(f.get('test_class_targets'))
        
#%% Keras imports and initializations
# Weights initializations
# bias are initailized as 0
import keras.backend as K
K.set_learning_phase(1)
from keras.layers import Input, Cropping2D, Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Model
from keras.initializers import RandomNormal

#conv_initG = RandomNormal(0, 0.05)
conv_initG = 'glorot_uniform'
#conv_initD = RandomNormal(0, 0.05)
conv_initD = 'he_normal'
gamma_init = RandomNormal(1., 0.05) # for batch normalization

def batchnorm():
    return BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)

#%% Generator Model
def GeneratorModel(input_shape):
    lay_input = Input(shape=input_shape,name='input_layer')
    
    padamt = 1
    crop = Cropping2D(cropping=((0, padamt), (0, padamt)), data_format=None)(lay_input)
    filtnum = 32
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initG,
                       name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initG,
                       name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,3):
        lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initG,
                       name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initG,
                       name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=2
    lay_deconv1 = Conv2D(filtnum*dd,(1,1),padding='same',kernel_initializer=conv_initG,
                       name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filtnum*dd,(3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filtnum*dd, (3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filtnum*dd, (3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filtnum*dd,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)
    
    lay_cleanup = Conv2DTranspose(filtnum*dd, (3, 3),kernel_initializer=conv_initG,
                       name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(filtnum*dd, (3,3), padding='same',kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # expanding blocks 2-1
    expnums = list(range(1,2))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(filtnum*dd,(1,1),padding='same',kernel_initializer=conv_initG,
                       name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(filtnum*dd,(3,3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(filtnum*dd, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(filtnum*dd, (3, 3),padding='same',kernel_initializer=conv_initG,
                       name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filtnum*dd,(1,1),padding='valid',kernel_initializer=conv_initG,
                       use_bias=False,name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(filtnum*dd, (3, 3),kernel_initializer=conv_initG,
                       name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(filtnum*dd, (3,3), padding='same',kernel_initializer=conv_initG,
                       use_bias=False,name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # regressor    
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
    lay_reg = Conv2D(1,(1,1), activation='linear',kernel_initializer=conv_initG,
                       name='regression')(lay_pad)    
    return Model(lay_input,lay_reg)
#%% Discriminator model
from keras.layers import Flatten, Dense#, Activation

def DiscriminatorModel(input_shape,test_shape,filtnum=16):
    # Conditional Inputs
    lay_input = Input(shape=input_shape,name='conditional_input')
    
    lay_step = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       name='StepdownLayer')(lay_input)
    
    usebias = False
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}'.format(rr))(lay_step)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}'.format(rr))(lay_step)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}'.format(rr))(lay_step)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}'.format(rr))(lay_merge)
#    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}'.format(rr))(lay_act)
    lay_act1 = LeakyReLU(alpha=0.2,name='leaky{}_2'.format(rr))(lay_stride)
    
    # Testing Input block
    lay_test = Input(shape=test_shape,name='test_input')
    lay_step2 = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='StepdownLayer2')(lay_test)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}t'.format(rr))(lay_step2)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}t'.format(rr))(lay_step2)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}t'.format(rr))(lay_step2)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}t'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}t'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}t'.format(rr))(lay_merge)
#    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1t'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}t'.format(rr))(lay_act)
    lay_act2 = LeakyReLU(alpha=0.2,name='leaky{}_2t'.format(rr))(lay_stride)
    
    # Merge blocks
    lay_act = concatenate([lay_act1,lay_act2],name='InputMerge')
    # contracting blocks 2-5
    for rr in range(2,6):
        lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',kernel_initializer=conv_initD,
                       use_bias=usebias,name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvAll_{}'.format(rr))(lay_merge)
#        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2,name='leaky{}_1'.format(rr))(lay_conv_all)
        lay_stride = Conv2D(filtnum*rr,(4,4),padding='valid',strides=(2,2),kernel_initializer=conv_initD,
                       use_bias=usebias,name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = LeakyReLU(alpha=0.2,name='leaky{}_2'.format(rr))(lay_stride)
    
    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1,name='Dense1')(lay_flat)
#    lay_sig = Activation('sigmoid',name='FinalAct')(lay_dense)
    
    return Model(inputs=[lay_input,lay_test],outputs=[lay_dense])

#%% prepare model for training
print("Generating models...")
from keras.optimizers import Adam
# set learning rates and parameters
lrD = 2e-4
lrG = 2e-4
λ = 10  # grad penalty weighting

# create models
DisModel = DiscriminatorModel(x_train.shape[1:],y_train.shape[1:],16)
GenModel = GeneratorModel(x_train.shape[1:])

# get tensors of inputs and outputs
real_A = GenModel.input
fake_B = GenModel.output
real_B = DisModel.inputs[1]
output_D_real = DisModel([real_A, real_B])
output_D_fake = DisModel([real_A, fake_B])
# create mixed output for gradient penalty
ep_input = K.placeholder(shape=(None,1,1,1))
mixed_B = Input(shape=x_train.shape[1:],
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

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_G = -loss_D_fake + 5 * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(GenModel.trainable_weights,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_G, loss_L1], training_updates)

netG_eval = K.function([real_A, real_B],[loss_L1])
netD_eval = K.function([real_A,real_B],[output_D_real])
netD_evalF = K.function([real_A],[output_D_fake])

#%% training
print('Starting training...')
ex_ind = 65
numIter = 10000
progstep = 100
valstep = 250
b_s = 8
val_b_s = 8
train_rat = 3
dis_loss = np.zeros((numIter,1))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),1))
templosses = np.zeros(np.int(x_val.shape[0]/val_b_s))


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
        batch_inds = np.random.randint(0,x_train.shape[0], size=b_s)
        cond_batch = x_train[batch_inds,...]
        real_batch = y_train[batch_inds,...]
        # train discrimator
        ϵ = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = netD_train([cond_batch, real_batch, ϵ])
    dis_loss[ii] = errD
    # Train Generator
    errG = netG_train([cond_batch, real_batch])
    gen_loss[ii] = errG
    if ii % progstep == 0:
        cond_samp = x_test[ex_ind,...][np.newaxis,...]
        simfulldose_im = GenModel.predict(cond_samp)[0,...,0]
        fulldose_im = y_test[ex_ind,...,0]
        samp_im = np.c_[cond_samp[0,...,0],simfulldose_im,fulldose_im]
        progress_ims[gg] = samp_im
        gg += 1
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,x_val.shape[0]))
            cond_batch = x_val[val_inds,...]
            real_batch = y_val[val_inds,...]
            valL1 = netG_eval([cond_batch,real_batch])[0]
            templosses[bb] = valL1
        cur_val_loss = np.mean(templosses)
        if cur_val_loss < np.min(val_loss):
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss))
        GenModel.save(model_filepath,True,False)
        tqdm.write("Saved model to file")
            
        val_loss[vv] = cur_val_loss           
        vv +=1
        
    t.set_postfix(Dloss=dis_loss[ii,0], Gloss=gen_loss[ii,0], L1loss = gen_loss[ii,1])
    
t.close()
del t

print('Training complete')

# display loss
from scipy.signal import medfilt
fig2 = plt.figure(2)
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],51),
         np.arange(numIter),medfilt(gen_loss[:,0],51),
         np.arange(numIter),medfilt(100*gen_loss[:,1],51),
         np.arange(0,numIter,valstep),100*val_loss)
plt.legend(['-Discriminator Loss',
            'Generator Loss',
            '100x L1 loss',
            '100x Validation L1 loss'])
plt.ylim([0,100])
plt.show()

#%%
print('Generating samples')
from keras.models import load_model
#K.set_learning_phase(0)
GenModel = load_model(model_filepath,None,False)

# get generator results
pr_bs = np.minimum(16,x_test.shape[0])
time1 = time.time()
test_output = GenModel.predict(x_test,batch_size=pr_bs)
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
multi_slice_viewer0(np.c_[x_test[...,0],test_output[...,0],y_test[...,0]],'Test Images',SSIMs)
