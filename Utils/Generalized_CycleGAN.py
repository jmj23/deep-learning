# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
# progress bar
from tqdm import tqdm, trange
# these are my .py scripts in Utils
import Models
from VisTools import multi_slice_viewer0
# for gifs
import imageio
# for comparing images
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

# A is the input image type
# B is the target image type

# Model Save Path/name
model_filepath = 'CyleGAN_model.h5'

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
C = 500  # Cycle Loss weighting

# Discriminator parameters
# number of downsampling blocks
numBlocks_D = 3
# initial filter number
numFilters_D = 32

# Generator Parameters
# number of downsampling blocks
numBlocks_G = 4
# initial filter number
numFilters_G = 16
# number of blocks that have strided convolution
noStride = 2
# Seem to have better results with batch norm off
use_bn = False
# True for regression output
# False for discrete output
A_reg = True
B_reg = False

# whether to plot progress images
plot_prog = True
# whether to save progress gif after training
gif_prog = True
# index of testing slice to use as progress image
ex_ind = 136
# number of iterations (batches) to train
numIter = 500
# how many iterations between updating progress image
progstep = 10
# how many iterations between checking validation score
# and saving model
valstep = 100
# batch size
b_s = 8
# validation batch size
val_b_s = 8
# training ratio between discriminator and generator
train_rat = 3

#%% Loading data
# Load training, validation, and testing data
# must be in the form [slices,x,y,channels]
if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        test_A = np.array(f.get('A_test'))
        test_B = np.array(f.get('B_test'))
        if training:
            train_A = np.array(f.get('A_train'))
            train_B = np.array(f.get('B_train'))
            val_A = np.array(f.get('A_val'))
            val_B = np.array(f.get('B_val'))
        
#%% Setup models for training
print("Generating models...")
from keras.optimizers import Adam
import keras.backend as K

# Discriminator models
# A is input
# B is target
# Cycle loss is between input A to recovered A
DisModel_A = Models.CycleGANdiscriminator(train_A.shape[1:],numFilters_D,numBlocks_D)
DisModel_B = Models.CycleGANdiscriminator(train_B.shape[1:],numFilters_D,numBlocks_D)

# Generator Models
GenModel_A2B = Models.CycleGANgenerator(train_A.shape[1:],train_B.shape[-1],
                                          numFilters_G,numBlocks_G,noStride,use_bn,B_reg)
GenModel_B2A = Models.CycleGANgenerator(train_B.shape[1:],train_A.shape[-1],
                                          numFilters_G,numBlocks_G,noStride,use_bn,A_reg)

# Endpoints of graph- A to B
real_A = GenModel_A2B.inputs[0]
fake_B = GenModel_A2B.outputs[0]
rec_A = GenModel_B2A([fake_B])
# Endpoints of graph- B to A
real_B = GenModel_B2A.inputs[0]
fake_A = GenModel_B2A.outputs[0]
rec_B = GenModel_A2B([fake_A])
# A to B and back generator function
fn_genB = K.function([real_A],[fake_B,rec_A])

# Discriminator scores
realBscore = DisModel_B([real_B])
fakeBscore = DisModel_B([fake_B])
realAscore = DisModel_A([real_A])
fakeAscore = DisModel_A([fake_A])

#%% B discriminator loss funBion
# create mixed output for gradient penalty
from keras.layers import Input
ep_input1 = K.placeholder(shape=(None,1,1,1))
mixed_B = Input(shape=train_B.shape[1:],
                    tensor=ep_input1 * real_B + (1-ep_input1) * fake_B)
mixedBscore = DisModel_B([mixed_B])
# discriminator losses
realBloss = K.mean(realBscore)
fakeBloss = K.mean(fakeBscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedBscore],[mixed_B])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_B = fakeBloss - realBloss + λ * grad_penalty

#%% A to B generator loss function
# Adversarial Loss + A-B-A cycle loss
loss_A2B = -fakeBloss
loss_A2B2A = K.mean(K.abs(rec_A-real_A))

#%% A discriminator loss function
# create mixed output for gradient penalty
ep_input2 = K.placeholder(shape=(None,1,1,1))
mixed_A = Input(shape=train_A.shape[1:],
                    tensor=ep_input2 * real_A + (1-ep_input2) * fake_A)
mixedAscore = DisModel_A([mixed_A])
# discriminator losses
realAloss = K.mean(realAscore)
fakeAloss = K.mean(fakeAscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedAscore],[mixed_A])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_A = fakeAloss - realAloss + λ * grad_penalty

#%% B to A generator loss function
# Adversarial Loss + A-B-A cycle loss
loss_B2A = -fakeAloss
loss_B2A2B = K.mean(K.abs(rec_B-real_B))

#%% Training functions
# Discriminator training function
loss_D = loss_B + loss_A
weights_D = DisModel_B.trainable_weights + DisModel_A.trainable_weights
D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(weights_D,[],loss_D)
fn_trainD = K.function([real_A, real_B, ep_input1, ep_input2],[loss_B, loss_A], D_trups)

# Generator Training funBion
loss_G = loss_A2B + C*loss_A2B2A + loss_B2A + C*loss_B2A2B
weights_G = GenModel_A2B.trainable_weights + GenModel_B2A.trainable_weights
A2B_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(weights_G,[], loss_G)
# Generator training funBion returns A2B discriminator loss and A2B2A cycle loss
fn_trainG = K.function([real_A,real_B], [loss_A2B,loss_A2B2A], A2B_trups)

# validation evaluate funBion
fn_evalCycle = K.function([real_A],[loss_A2B2A])

#%% Progress plotting function
def ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax):
    A_samp = test_A[ex_ind,...][np.newaxis,...]
    [Btest,rec] = fn_genB([A_samp])
    if B_reg:
        Binds = np.argmax(Btest[0],axis=-1)
        B_current = Binds/Btest.shape[-1]
        Binds_truth = np.argmax(test_B[ex_ind],axis=-1)
        B_truth = Binds_truth/Btest.shape[-1]
    else:
        B_current = Btest[0,...,0]
        B_truth = test_B[ex_ind,...,0]
    if A_reg:
        Ainds = np.argmax(A_samp[0],axis=-1)
        A_current = Ainds/test_A.shape[-1]
        Ainds_rec = np.argmax(rec[0],axis=-1)
        A_rec = Ainds_rec/rec.shape[-1]
    else:
        A_current = A_samp[0,...,0]
        A_rec= rec[0,...,0]    
        
    samp_im1 = np.c_[A_current,B_current]
    samp_im2 = np.c_[A_rec,B_truth]
    samp_im = np.r_[samp_im1,samp_im2]
    ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
    ax.set_axis_off()
    ax.set_clip_box([0,1])
    ax.title('Current training state')
    plt.pause(.001)
    plt.draw()
    return samp_im

#%% training
print('Starting training...')

# preallocate for the training and validation losses
dis_loss = np.zeros((numIter,2))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),2))
templosses = np.zeros((np.int(val_A.shape[0]/val_b_s),2))

# Updatable plot
if plot_prog:
    plt.ion()
    fig, ax = plt.subplots()
    _ = ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax)
    # preallocate image display
    progress_ims = np.zeros((np.int(numIter/progstep),2*256,2*256))
    gg = 0


print('Training adversarial model')

# validation counter
vv = 0

t = trange(numIter,file=sys.stdout)
for ii in t:
    for _ in range(train_rat):
        # Train Discriminator
        # grab random training samples
        batch_inds = np.random.choice(train_A.shape[0], b_s, replace=False)
        A_batch = train_A[batch_inds,...]
        batch_inds = np.random.choice(train_B.shape[0], b_s, replace=False)
        B_batch = train_B[batch_inds,...]
        # train discrimator
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        ϵ2 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([A_batch, B_batch, ϵ1,ϵ2])
    dis_loss[ii] = errD
    # Train Generator
    errG = fn_trainG([A_batch, B_batch])
    gen_loss[ii] = errG
    if plot_prog:
        if ii % progstep == 0:
            # progress image plotting
            samp_im = ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax)
            progress_ims[gg] = samp_im
            gg += 1
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,val_A.shape[0]))
            A_batch = val_A[val_inds,...]
            valL1 = fn_evalCycle([A_batch])[0]
            templosses[bb] = valL1
        
        cur_val_loss = np.mean(templosses,axis=0)
        if cur_val_loss[0] <= np.min(val_loss,axis=0)[0]:
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss[0]))
            GenModel_A2B.save(model_filepath.format('A2B'),True,False)
            GenModel_B2A.save(model_filepath.format('B2A'),True,False)
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
model_json = GenModel_A2B.to_json()
with open("BackupModel_A2B.json", "w") as json_file:
    json_file.write(model_json)
GenModel_A2B.save_weights("BackupModel_A2B.h5")

model_json = GenModel_B2A.to_json()
with open("BackupModel_B2A.json", "w") as json_file:
    json_file.write(model_json)
GenModel_B2A.save_weights("BackupModel_B2A.h5")

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
plt.legend(['-Discriminator B Loss',
            '-Discriminator A Loss',
            '1000x Cycle Loss',
            '1000x Validation Cycle loss'])
plt.ylim([0,60])
plt.show()

#%%
print('Generating samples')
from keras.models import load_model

if False:
    # Load backup models
    from keras.models import model_from_json
    json_file = open('BackupModel_A2B.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    GenModel_A2B = model_from_json(loaded_model_json)
    GenModel_A2B.load_weights("BackupModel_A2B.h5")
    from keras.models import model_from_json
    json_file = open('BackupModel_B2A.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    GenModel_B2A = model_from_json(loaded_model_json)
    GenModel_B2A.load_weights("BackupModel_B2A.h5")
    print("Loaded backup models from weights")
else:
    # Load regular checkpointed models
    GenModel_A2B = load_model(model_filepath.format('A2B'),None,False)
    GenModel_B2A = load_model(model_filepath.format('B2A'),None,False)

# get generator results
time1 = time.time()
Btest = GenModel_A2B.predict(test_A)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/test_A.shape[0],' ms per slice')

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

if B_reg:
    # Calculate SSIM between test images
    SSIMs = [ssim(im1,im2) for im1, im2 in zip(test_B[...,0],Btest[...,0])]
    print('Mean SSIM of ', np.mean(SSIMs))
    print('SSIM range of ', np.round(np.min(SSIMs),3), ' - ', np.round(np.max(SSIMs),3))

# Display test images in grid
# Input A     | Output B
#----------------------
# Recovered A | Actual B

Arec = GenModel_B2A.predict(Btest)

if B_reg:
    Binds = np.argmax(Btest,axis=-1)
    B_output = Binds/Btest.shape[-1]
    Binds_truth = np.argmax(test_B,axis=-1)
    B_truth = Binds_truth/Btest.shape[-1]
else:
    B_output = Btest[...,0]
    B_truth = test_B[...,0]
if A_reg:
    Ainds = np.argmax(test_A,axis=-1)
    A_output = Ainds/test_A.shape[-1]
    Ainds_rec = np.argmax(Arec,axis=-1)
    A_rec = Ainds_rec/Arec.shape[-1]
else:
    A_output = test_A[...,0]
    A_rec= Arec[...,0]
        
disp_im1 = np.c_[A_output,B_output]
disp_im2 = np.c_[A_rec,B_truth]
test_disp = np.concatenate((disp_im1,disp_im2),axis=1)

if B_reg:
    multi_slice_viewer0(test_disp,'Test Images',SSIMs)
else:
    multi_slice_viewer0(test_disp,'Test Images')