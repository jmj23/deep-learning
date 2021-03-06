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

#%%
# Model Save Path/name
model_filepath = 'DeepMuMapCyleGAN_{}_model.h5'

# Data path/name
datapath = 'CycleGAN_data_TVT.hdf5'

# Load training, validation, and testing data
# must be in the form [slices,x,y,channels]
if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        test_MR = np.array(f.get('MR_test'))
        test_CT = np.array(f.get('CT_test_con'))
        train_MR = np.array(f.get('MR_train'))
        train_CT = np.array(f.get('CT_train_con'))
        val_MR = np.array(f.get('MR_val'))
        val_CT = np.array(f.get('CT_val_con'))
        
#%% Setup models for training
print("Generating models...")
from keras.optimizers import Adam
import keras.backend as K
# set learning rates and parameters
lrD = 1e-4
lrG = 1e-4
λ = 10  # grad penalty weighting- leave alone
C = 500  # Cycle Loss weighting

# Discriminator models
# CT is target
# MR is input
# Cycle loss is between MR to recovered MR

# number of downsampling blocks
numBlocks = 3
# initial filter number
numFilters = 32

DisModel_CT = Models.CycleGANdiscriminator(train_CT.shape[1:],numFilters,numBlocks)
DisModel_MR = Models.CycleGANdiscriminator(train_MR.shape[1:],numFilters,numBlocks)
# Generator Models
# number of downsampling blocks
numBlocks = 4
# initial filter number
numFilters = 16
# number of blocks that have strided convolution
noStride = 2
# Seem to have better results with batch norm off
use_bn = False

GenModel_MR2CT = Models.CycleGANgenerator(train_MR.shape[1:],train_CT.shape[-1],numFilters,numBlocks,noStride,use_bn)
GenModel_CT2MR = Models.CycleGANgenerator(train_CT.shape[1:],train_MR.shape[-1],numFilters,numBlocks,noStride,use_bn)

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
loss_G = loss_MR2CT + C*loss_MR2CT2MR + loss_CT2MR + C*loss_CT2MR2CT
weights_G = GenModel_MR2CT.trainable_weights + GenModel_CT2MR.trainable_weights
MR2CT_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], loss_G)
# Generator training function returns MR2CT discriminator loss and MR2CT2MR cycle loss
fn_trainG = K.function([real_MR,real_CT], [loss_MR2CT,loss_MR2CT2MR], MR2CT_trups)

# validation evaluate function
fn_evalCycle = K.function([real_MR],[loss_MR2CT2MR])

#%% training
print('Starting training...')
# index of testing set to use as progress image
ex_ind = 136
# number of iterations (batches) to train
numIter = 25000
# how many steps between creating progress image
progstep = 100
# how many steps between checking validation score
# and saving model
valstep = 50
# batch size
b_s = 8
# validation batch size
val_b_s = 8
# training ratio between discriminator and generator
train_rat = 3
# preallocate for the training and validation losses
dis_loss = np.zeros((numIter,2))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(val_MR.shape[0]/val_b_s),2))

# Updatable plot
plt.ion()
fig, ax = plt.subplots()
MR_samp = test_MR[ex_ind,...][np.newaxis,...]
[test,rec] = fn_genCT([MR_samp])
samp_im1 = np.c_[MR_samp[0,...,0],test[0,...,0]]
samp_im2 = np.c_[rec[0,...,0],test_CT[ex_ind,...,0]]
samp_im = np.r_[samp_im1,samp_im2]
ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
ax.set_axis_off()
ax.set_clip_box([0,1])
plt.pause(.001)
plt.draw()


print('Training adversarial model')
# preallocate image display
progress_ims = np.zeros((np.int(numIter/progstep),2*256,2*256))
gg = 0
vv = 0

t = trange(numIter,file=sys.stdout)
for ii in t:
    for _ in range(train_rat):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(train_MR.shape[0], b_s, replace=False)
        MR_batch = train_MR[batch_inds,...]
        batch_inds = np.random.choice(train_CT.shape[0], b_s, replace=False)
        CT_batch = train_CT[batch_inds,...]
        # train discrimator
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        ϵ2 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([MR_batch, CT_batch, ϵ1,ϵ2])
    dis_loss[ii] = errD
    # Train Generator
    errG = fn_trainG([MR_batch, CT_batch])
    gen_loss[ii] = errG
    if ii % progstep == 0:
        # progress image plotting
        MR_samp = test_MR[ex_ind,...][np.newaxis,...]
        [test,rec] = fn_genCT([MR_samp])
        samp_im1 = np.c_[MR_samp[0,...,0],test[0,...,0]]
        samp_im2 = np.c_[rec[0,...,0],test_CT[ex_ind,...,0]]
        samp_im = np.r_[samp_im1,samp_im2]
        progress_ims[gg] = samp_im
        ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
        plt.pause(.001)
        plt.draw()
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
fig5 = plt.figure(5)
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],5),
         np.arange(numIter),-medfilt(dis_loss[:,1],5),
         np.arange(numIter),medfilt(1000*gen_loss[:,1],5),
         np.arange(0,numIter,valstep),1000*val_loss[:,0])
plt.legend(['-Discriminator CT Loss',
            '-Discriminator MR Loss',
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
#json_file = open('BackupModel_MR2CT.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#MR2CT_Model = model_from_json(loaded_model_json)
#MR2CT_Model.load_weights("BackupModel_MR2CT.h5")
# from keras.models import model_from_json
#json_file = open('BackupModel_CT2MR.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#CT2MR_Model = model_from_json(loaded_model_json)
#CT2MR_Model.load_weights("BackupModel_CT2MR.h5")
#print("Loaded backup models from weights")
#~#~#~#~#~#~#~#~#~#~#~#~#~#

GenModel_MR2CT = load_model(model_filepath.format('MR2CT'),None,False)
GenModel_CT2MR = load_model(model_filepath.format('CT2MR'),None,False)

# get generator results
time1 = time.time()
CTtest = GenModel_MR2CT.predict(test_MR)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/test_MR.shape[0],' ms per slice')

# Display progress images, if they exist

if 'progress_ims' in locals():
    multi_slice_viewer0(progress_ims,'Training Progress Images')
    
    # save progress ims to gif
    output_file = 'ProgressIms.gif'
    gif_ims = np.copy(progress_ims)
    gif_ims[gif_ims<0] = 0
    gif_ims[gif_ims>1] = 1
    gif_ims = (255*gif_ims).astype(np.uint8)
    images = [gif_ims[ii,...] for ii in range(gif_ims.shape[0])]
    imageio.mimsave(output_file, images, duration=1/6,loop=1)

# Calculate SSIM between test images
SSIMs = [ssim(im1,im2) for im1, im2 in zip(test_CT[...,0],CTtest[...,0])]
print('Mean SSIM of ', np.mean(SSIMs))
print('SSIM range of ', np.round(np.min(SSIMs),3), ' - ', np.round(np.max(SSIMs),3))

# Display test images in grid
# Input MR     | Output CT
#----------------------
# Recovered MR | Actual CT

MRrec = GenModel_CT2MR.predict(CTtest)
disp1 = np.c_[test_MR[...,0],CTtest[...,0]]
disp2 = np.c_[MRrec[...,0],test_CT[...,0]]
test_disp = np.concatenate((disp1,disp2),axis=1)
multi_slice_viewer0(test_disp,'Test Images',SSIMs)

# Load full MR data and run inferencing for visual inspection
with h5py.File('CycleGAN_FullMRdata.hdf5','r') as f:
        full_MR = np.array(f.get('MR_full'))
full_CT = GenModel_MR2CT.predict(full_MR)
full_MRrec = GenModel_CT2MR.predict(full_CT)
full_disp = np.concatenate((full_MR[...,0],full_CT[...,0],full_MRrec[...,0]),axis=2)
multi_slice_viewer0(full_disp,'Full Images')
