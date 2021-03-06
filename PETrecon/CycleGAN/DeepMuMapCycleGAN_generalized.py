# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from tqdm import tqdm, trange
import Models
import imageio
from VisTools import multi_slice_viewer0
from skimage.measure import compare_ssim as ssim

# Use first available GPU
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

np.random.seed(seed=1)
    
#%%~#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Parameters/variables to set
#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

# Model Save Path/name
timestr = time.strftime("%Y%m%d%H%M")
model_filepath = 'DeepMuMapCycleGAN_{}_model_{}.h5'.format('{}',timestr)
# load previous model
model_filepath = 'DeepMuMapCyleGAN_MR2CT_model_201805031054.h5'

# Data path/name
datapath = 'CycleGAN_data_TVT.hdf5'

# Training or testing
# False will only load the testing
# datset
training = True

# set learning rates and loss weights
lrD = 1e-5  # discriminator learning rate
lrG = 1e-5  # generator learning rate
λ = 10  # grad penalty weighting- leave alone
C = 200  # Cycle Loss weighting

# Discriminator parameters
# number of downsampling blocks
numBlocks_D = 3
# initial filter number
numFilters_D = 16

# Generator Parameters
# number of downsampling blocks
numBlocks_G = 4
# initial filter number
numFilters_G = 16
# number of blocks that have strided convolution
noStride = 4
# Seem to have better results with batch norm off
use_bn = False
# True for regression output
# False for discrete output
CT_reg = True
MR_reg = True

# whether to plot progress images
plot_prog = False
# whether to save progress gif after training
gif_prog = False
# whether to just save loss plots rather than
# displaying them
savenodisplay = True
# index of testing slice to use as progress image
ex_ind = 136
# number of iterations (batches) to train
numIter = 80000
# how many iterations between updating progress image
progstep = 100
# how many iterations between checking validation score
# and saving model
valstep = 100
# batch size
b_s = 16
# validation batch size
val_b_s = 16
# training ratio between discriminator and generator
train_rat = 5
# whether to use learning rate decay
LR_decay = False
LR_period = np.int(numIter/5)


# Whether or not to pretrain generators
# Leave as false if data is not roughly 
# aligned and in order
# Can turn pre training on or off with
# command line argument
# or just set the default
# example: $ python DeepMuMapCycleGAN_generalized.py True
if len(sys.argv) > 1:
    L1pretrain = sys.argv[1].lower() == 'true'
else:
    # default pretrain setting if no argument provided    
    L1pretrain = True
    
Dpretrain = True
pretrain_epochs = 5
pretrain_lr = 1e-4

    

#%% Loading data
# Load training, validation, and testing data
# must be in the form [slices,x,y,channels]
if not 'test_MR' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        test_MR = np.array(f.get('MR_test'))
        test_CT = np.array(f.get('CT_test_con'))
        if training:
            train_MR = np.array(f.get('MR_train'))
            train_CT = np.array(f.get('CT_train_con'))
            val_MR = np.array(f.get('MR_val'))
            val_CT = np.array(f.get('CT_val_con'))
    print('Data Loaded')
        
#%% Setup models for training
print("Creating models...")
from keras.optimizers import Adam
import keras.backend as K

# Discriminator models
# CT is target
# MR is input
# Cycle loss is between MR to recovered MR
DisModel_CT = Models.CycleGANdiscriminator(train_CT.shape[1:],numFilters_D,numBlocks_D)
DisModel_MR = Models.CycleGANdiscriminator(train_MR.shape[1:],numFilters_D,numBlocks_D)

# Generator Models
GenModel_MR2CT = Models.CycleGANgenerator(train_MR.shape[1:],train_CT.shape[-1],
                                          numFilters_G,numBlocks_G,noStride,use_bn,CT_reg)
GenModel_CT2MR = Models.CycleGANgenerator(train_CT.shape[1:],train_MR.shape[-1],
                                          numFilters_G,numBlocks_G,noStride,use_bn,MR_reg)

# Endpoints of graph- MR to CT
real_MR = GenModel_MR2CT.inputs[0]
fake_CT = GenModel_MR2CT.outputs[0]
rec_MR = GenModel_CT2MR([fake_CT])
# Endpoints of graph- CT to MR
real_CT = GenModel_CT2MR.inputs[0]
fake_MR = GenModel_CT2MR.outputs[0]
rec_CT = GenModel_MR2CT([fake_MR])
# MR to CT and back generator function
fn_genCT = K.function([real_MR],[fake_CT,rec_MR])
fn_gen_pre = K.function([real_MR,real_CT],[fake_CT,rec_MR])

# Discriminator scores
realCTscore = DisModel_CT([real_CT])
fakeCTscore = DisModel_CT([fake_CT])
realMRscore = DisModel_MR([real_MR])
fakeMRscore = DisModel_MR([fake_MR])

#%% CT discriminator loss function
# create mixed output for gradient penalty
from keras.layers import Input
ep_input1 = K.placeholder(shape=(None,1,1,1))
mixed_CT = Input(shape=train_CT.shape[1:],
                    tensor=ep_input1 * real_CT + (1-ep_input1) * fake_CT)
mixedCTscore = DisModel_CT([mixed_CT])
# discriminator losses
realCTloss = K.mean(realCTscore)
fakeCTloss = K.mean(fakeCTscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedCTscore],[mixed_CT])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_CT = fakeCTloss - realCTloss + λ * grad_penalty

#%% MR to CT generator loss function
# Adversarial Loss + MR-CT-MR cycle loss
loss_MR2CT = -fakeCTloss
loss_MR2CT2MR = K.mean(K.abs(rec_MR-real_MR))

#%% MR discriminator loss function
# create mixed output for gradient penalty
ep_input2 = K.placeholder(shape=(None,1,1,1))
mixed_MR = Input(shape=train_MR.shape[1:],
                    tensor=ep_input2 * real_MR + (1-ep_input2) * fake_MR)
mixedMRscore = DisModel_MR([mixed_MR])
# discriminator losses
realMRloss = K.mean(realMRscore)
fakeMRloss = K.mean(fakeMRscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedMRscore],[mixed_MR])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_MR = fakeMRloss - realMRloss + λ * grad_penalty

#%% CT to MR generator loss function
# Adversarial Loss + MR-CT-MR cycle loss
loss_CT2MR = -fakeMRloss
loss_CT2MR2CT = K.mean(K.abs(rec_CT-real_CT))

#%% Training functions
# Discriminator training function
loss_D = loss_CT + loss_MR
weights_D = DisModel_CT.trainable_weights + DisModel_MR.trainable_weights

D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(weights_D,[],loss_D)
fn_trainD = K.function([real_MR, real_CT, ep_input1, ep_input2],[loss_CT, loss_MR], D_trups)

# Generator Training function
loss_G = loss_MR2CT + C*loss_MR2CT2MR + loss_CT2MR #+ C*loss_CT2MR2CT
weights_G = GenModel_MR2CT.trainable_weights + GenModel_CT2MR.trainable_weights

MR2CT_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], loss_G)
# Generator training function returns MR2CT discriminator loss and MR2CT2MR cycle loss

fn_trainG = K.function([real_MR,real_CT], [loss_MR2CT,loss_MR2CT2MR], MR2CT_trups)

# validation evaluate function
fn_evalCycle = K.function([real_MR],[loss_MR2CT2MR])

#%% Progress plotting function
def ProgressPlot(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax):
    MR_samp = test_MR[ex_ind,...][np.newaxis,...]
    [CTtest,rec] = fn_genCT([MR_samp])
    if CT_reg:
        CT_current = CTtest[0,...,0]
        CT_truth = test_CT[ex_ind,...,0]
    else:
        CTinds = np.argmax(CTtest[0],axis=-1)
        CT_current = CTinds/CTtest.shape[-1]
        CTinds_truth = np.argmax(test_CT[ex_ind],axis=-1)
        CT_truth = CTinds_truth/CTtest.shape[-1]
    if MR_reg:
        MR_current = MR_samp[0,...,0]
        MR_rec= rec[0,...,0]  
    else:  
        MRinds = np.argmax(MR_samp[0],axis=-1)
        MR_current = MRinds/test_MR.shape[-1]
        MRinds_rec = np.argmax(rec[0],axis=-1)
        MR_rec = MRinds_rec/rec.shape[-1]
        
    samp_im1 = np.c_[MR_current,CT_current]
    samp_im2 = np.c_[MR_rec,CT_truth]
    samp_im = np.r_[samp_im1,samp_im2]
    ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
    ax.set_axis_off()
    ax.set_clip_box([0,1])
    ax.set_title('Current training state')
    plt.text(-0.02, .96,'MR input', ha='right', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(-0.02, 0.04,'Recovered MR', ha='right', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(1.02, 0.04,'CT truth', ha='left', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(1.02, 0.96,'CT predict', ha='left', va='top',color='red',
                      transform=ax.transAxes)
    plt.pause(.001)
    plt.draw()
    return samp_im

#%% Progress plotting function for pre-training
def ProgressPlotPretrain(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax):
    MR_samp = test_MR[ex_ind,...][np.newaxis,...]
    CT_samp = test_CT[ex_ind,...][np.newaxis,...]
    [CTpre,MRpre] = fn_gen_pre([MR_samp,CT_samp])
    if CT_reg:
        CT_pre = CTpre[0,...,0]
        CT_truth = CT_samp[0,...,0]
    else:
        CTinds = np.argmax(CTpre[0],axis=-1)
        CT_pre = CTinds/CTpre.shape[-1]
        CTinds_truth = np.argmax(CT_samp[ex_ind],axis=-1)
        CT_truth = CTinds_truth/CT_samp.shape[-1]
    if MR_reg:
        MR_truth = MR_samp[0,...,0]
        MR_pre = MRpre[0,...,0]  
    else:  
        MRinds = np.argmax(MR_samp[0],axis=-1)
        MR_truth = MRinds/test_MR.shape[-1]
        MRinds_pre = np.argmax(MRpre[0],axis=-1)
        MR_pre = MRinds_pre/MRpre.shape[-1]
        
    samp_im1 = np.c_[MR_truth,MR_pre]
    samp_im2 = np.c_[CT_pre,CT_truth]
    samp_im = np.r_[samp_im1,samp_im2]
    ax.imshow(samp_im,cmap='gray',vmin=np.min(samp_im),vmax=np.max(samp_im))
    ax.set_axis_off()
    ax.set_clip_box([0,1])
    ax.set_title('Current pre-training state')
    plt.text(-0.02, .96,'MR input', ha='right', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(-0.02, 0.04,'CT predict', ha='right', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(1.02, 0.04,'CT input', ha='left', va='top',color='red',
                      transform=ax.transAxes)
    plt.text(1.02, 0.96,'MR recover', ha='left', va='top',color='red',
                      transform=ax.transAxes)
    plt.pause(.001)
    plt.draw()
    return samp_im

#%% Optional Pretraining using L1 loss
if L1pretrain:
    # Generator pretraining losses
    pretrain_loss_MR2CT = K.mean(K.abs(fake_CT-real_CT))
    pretrain_loss = pretrain_loss_MR2CT + loss_MR2CT2MR
    # Generator pretraining updates and function
    pretrain_trups = Adam(lr=pretrain_lr, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], pretrain_loss)
    pretrain_fn = K.function([real_MR,real_CT],[pretrain_loss_MR2CT,loss_MR2CT2MR],updates=pretrain_trups)
    # calculate equivalent batches for number of epochs
    pre_numIter = np.int(pretrain_epochs*train_MR.shape[0]/b_s)
    # Updatable plot
    if plot_prog:
        plt.ion()
        fig, ax = plt.subplots()
        ProgressPlotPretrain(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax)
        # preallocate image display
        numProgs = np.maximum(np.int(pre_numIter/progstep),1)
        pre_progress_ims = np.zeros((numProgs+1,2*256,2*256))
        pg = 0
    # pre allocate losses
    pre_dis_loss = np.zeros((pre_numIter,2))
    pre_gen_loss = np.zeros((pre_numIter,2))
    # setup progress bar
    pre_t = trange(pre_numIter,file=sys.stdout)
    # start pretraining
    tqdm.write('Pre-training...')
    for ii in pre_t:
        if Dpretrain:
            # pretrain discrimators
            for _ in range(train_rat):
                # get batches
                batch_inds = np.random.choice(train_MR.shape[0], b_s, replace=False)
                MR_batch = train_MR[batch_inds,...]
                CT_batch = train_CT[batch_inds,...]
                ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
                ϵ2 = np.random.uniform(size=(b_s, 1, 1 ,1))
                errD = fn_trainD([MR_batch, CT_batch, ϵ1,ϵ2])
            pre_dis_loss[ii] = errD
        else:
            # get batches
            batch_inds = np.random.choice(train_MR.shape[0], b_s, replace=False)
            MR_batch = train_MR[batch_inds,...]
            CT_batch = train_CT[batch_inds,...]
        # Pre train Generators
        errG = pretrain_fn([MR_batch, CT_batch])
        pre_gen_loss[ii] = errG
        # Update progress image
        if plot_prog and ii % progstep == 0:
            samp_im = ProgressPlotPretrain(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax)
            pre_progress_ims[pg] = samp_im
            pg += 1
        # Print generator losses in progress bar
        pre_t.set_postfix(Dis_Error=errD[0],M2CTR_Error= errG[0],Cycle_Error=errG[1])
        
    pre_t.close()
    del pre_t
    
    print('Pretraining complete')
    fig = plt.figure()
    plt.plot(np.arange(pre_numIter),-pre_dis_loss[:,0],
             np.arange(pre_numIter),-pre_dis_loss[:,1],
             np.arange(pre_numIter),100*pre_gen_loss[:,0],
             np.arange(pre_numIter),100*pre_gen_loss[:,1])
             
    plt.legend(['-Discriminator CT Loss',
                '-Discriminator MR Loss',
                '100x MR2CT L1 Loss',
                '100x Cycle loss'])
    plt.ylim([0,100])
    if savenodisplay:
        name = 'CycleGANpretrainLossPlot_{}.png'.format(timestr)
        plt.savefig(name)
    else:
        plt.show()
    
    
    # save pretraining
    name = 'CycleGANpretrainLosses_{}.txt'.format(timestr)
    data = (pre_dis_loss[:,0],pre_dis_loss[:,1],pre_gen_loss[:,0],pre_gen_loss[:,1])
    np.savetxt(name,data)

#%% training
print('Starting training...')

# preallocate for the training and validation losses
dis_loss = np.zeros((numIter,2))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(val_MR.shape[0]/val_b_s),2))

# Updatable plot
if plot_prog:
    plt.ion()
    fig, ax = plt.subplots()
    _ = ProgressPlot(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax)
    # preallocate image display
    numProgs = np.maximum(np.int(numIter/progstep),1)
    progress_ims = np.zeros((numProgs,2*256,2*256))
    gg = 0


print('Training adversarial model')

# validation counter
vv = 0
if 't' in locals():
    del t
t = trange(numIter,file=sys.stdout)
for ii in t:
    for _ in range(train_rat):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(train_MR.shape[0], b_s, replace=False)
        MR_batch = train_MR[batch_inds,...]
        batch_inds = np.random.choice(train_CT.shape[0], b_s, replace=False)
        CT_batch = train_CT[batch_inds,...]
        # train discrimators
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        ϵ2 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([MR_batch, CT_batch, ϵ1,ϵ2])
    dis_loss[ii] = errD
    # Train Generator
    errG = fn_trainG([MR_batch, CT_batch])
    gen_loss[ii] = errG
    if plot_prog and ii % progstep == 0:
        # progress image plotting
        samp_im = ProgressPlot(test_MR,test_CT,ex_ind,MR_reg,CT_reg,ax)
        progress_ims[gg] = samp_im
        gg += 1
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,val_MR.shape[0]))
            MR_batch = val_MR[val_inds,...]
            valL1 = fn_evalCycle([MR_batch])[0]
            templosses[bb] = valL1
        
        cur_val_loss = np.mean(templosses,axis=0)
        if cur_val_loss[0] <= np.min(val_loss,axis=0)[0]:
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss[0]))
            GenModel_MR2CT.save(model_filepath.format('MR2CT'),True,False)
            GenModel_CT2MR.save(model_filepath.format('CT2MR'),True,False)
            tqdm.write("Saved models to file")
        else:
            tqdm.write('Validation loss did not decrease: {:.02e}'.format(cur_val_loss[0]))
            
        val_loss[vv] = cur_val_loss           
        vv +=1
        
        # save losses
        name = 'CycleGANtrainLosses_{}.txt'.format(timestr)
        data = (dis_loss[:,0],dis_loss[:,1],gen_loss[:,0],gen_loss[:,1])
        np.savetxt(name,data)
        
    if LR_decay and (ii) % LR_period == 0 and ii>0:
        lrG = lrG/2
        MR2CT_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], loss_G)
        fn_trainG = K.function([real_MR,real_CT], [loss_MR2CT,loss_MR2CT2MR], MR2CT_trups)
        print('Updated generator learning rate to {:.3g}'.format(lrG))
        lrD = lrD/2
        D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(weights_D,[],loss_D)
        fn_trainD = K.function([real_MR, real_CT, ep_input1, ep_input2],[loss_CT, loss_MR], D_trups)
        print('Updated discriminator learning rate to {:.3g}'.format(lrD))
        
    t.set_postfix(Dloss=dis_loss[ii,0],CycleLoss = gen_loss[ii,1])
    
t.close()
del t

print('Training complete')

# backup model saving- writes as weights and network structure
# which is more reliable than saving model as single file
model_json = GenModel_MR2CT.to_json()
with open("BackupModel_MR2CT.json", "w") as json_file:
    json_file.write(model_json)
GenModel_MR2CT.save_weights("BackupModel_MR2CT.h5")

model_json = GenModel_CT2MR.to_json()
with open("BackupModel_CT2MR.json", "w") as json_file:
    json_file.write(model_json)
GenModel_CT2MR.save_weights("BackupModel_CT2MR.h5")

print('Backup Models saved')

# display loss
# smoothed since it's usually pretty rocky
# the L1 losses are multipled by 1000 to match scale
from scipy.signal import medfilt
fig = plt.figure()
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],5),
         np.arange(numIter),-medfilt(dis_loss[:,1],5),
         np.arange(numIter),medfilt(1000*gen_loss[:,1],5),
         np.arange(0,numIter,valstep),1000*val_loss[:,0])
plt.legend(['-Discriminator CT Loss',
            '-Discriminator MR Loss',
            '1000x Cycle Loss',
            '1000x Validation Cycle loss'])
plt.ylim([0,60])
if savenodisplay:
    name = 'CycleGANtrainLossPlot_{}.png'.format(timestr)
    plt.savefig(name)
else:
    plt.show()

#%%
print('Generating samples')
from keras.models import load_model

if False:
    # Load backup models
    from keras.models import model_from_json
    json_file = open('BackupModel_MR2CT.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    GenModel_MR2CT = model_from_json(loaded_model_json)
    GenModel_MR2CT.load_weights("BackupModel_MR2CT.h5")
    from keras.models import model_from_json
    json_file = open('BackupModel_CT2MR.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    GenModel_CT2MR = model_from_json(loaded_model_json)
    GenModel_CT2MR.load_weights("BackupModel_CT2MR.h5")
    print("Loaded backup models from weights")
else:
    # Load regular checkpointed models
    GenModel_MR2CT = load_model(model_filepath.format('MR2CT'),None,False)
    GenModel_CT2MR = load_model(model_filepath.format('CT2MR'),None,False)

# get generator results
time1 = time.time()
CTtest = GenModel_MR2CT.predict(test_MR)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/test_MR.shape[0],' ms per slice')

# Display progress images, if created
if plot_prog:
    if L1pretrain:
        progress_ims = np.concatenate((pre_progress_ims,progress_ims),axis=0)
        output_file = 'ProgressIms_w_pretrain_{}.gif'.format(timestr)
    else:
        output_file = 'ProgressIms_{}.gif'.format(timestr)
    multi_slice_viewer0(progress_ims,'Training Progress Images')
    
    # save progress ims to gif
    gif_ims = np.copy(progress_ims)
    gif_ims[gif_ims<0] = 0
    gif_ims[gif_ims>1] = 1
    gif_ims = (255*gif_ims).astype(np.uint8)
    images = [gif_ims[ii,...] for ii in range(gif_ims.shape[0])]
    imageio.mimsave(output_file, images, duration=1/6,loop=1)

if CT_reg:
    # Calculate SSIM between test images
    SSIMs = [ssim(im1,im2) for im1, im2 in zip(test_CT[...,0],CTtest[...,0])]
    print('Mean SSIM of ', np.mean(SSIMs))
    print('SSIM range of ', np.round(np.min(SSIMs),3), ' - ', np.round(np.max(SSIMs),3))

# Display test images in grid
# Input MR     | Output CT
#----------------------
# Recovered MR | Actual CT

MRrec = GenModel_CT2MR.predict(CTtest)

if CT_reg:
    CT_output = CTtest[...,0]
    CT_truth = test_CT[...,0]
else:
    CTinds = np.argmax(CTtest,axis=-1)
    CT_output = CTinds/CTtest.shape[-1]
    CTinds_truth = np.argmax(test_CT,axis=-1)
    CT_truth = CTinds_truth/CTtest.shape[-1]
if MR_reg:
    MR_output = test_MR[...,0]
    MR_rec= MRrec[...,0]
else:
    MRinds = np.argmax(test_MR,axis=-1)
    MR_output = MRinds/test_MR.shape[-1]
    MRinds_rec = np.argmax(MRrec,axis=-1)
    MR_rec = MRinds_rec/MRrec.shape[-1]
        
disp_im1 = np.c_[MR_output,CT_output]
disp_im2 = np.c_[MR_rec,CT_truth]
test_disp = np.concatenate((disp_im1,disp_im2),axis=1)

if CT_reg:
    multi_slice_viewer0(test_disp,'Test Images',SSIMs)
else:
    multi_slice_viewer0(test_disp,'Test Images')

# Load full MR data and run inferencing for visual inspection
#with h5py.File('CycleGAN_FullMRdata.hdf5','r') as f:
#        full_MR = np.array(f.get('MR_full'))
#full_CT = GenModel_MR2CT.predict(full_MR)
#full_MRrec = GenModel_CT2MR.predict(full_CT)
#full_disp = np.concatenate((full_MR[...,0],full_CT[...,0],full_MRrec[...,0]),axis=2)
#multi_slice_viewer0(full_disp,'Full Images')
