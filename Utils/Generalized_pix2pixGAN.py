# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: Jacob Johnson, MS
@email: jmjohnson33@wisc.edu
@url: https://github.com/jmj23/deep-learning/

All rights reserved
"""
import sys
import os
# Utility files need to be on the path
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from VisTools import multi_slice_viewer0 # for visualizing results
import JMJmodels # Where Generator and Discriminator models are defined

from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from tqdm import tqdm, trange
import imageio
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
model_filepath = 'Pix2pix_model_{}.h5'.format(timestr)
# load previous model
#model_filepath = 'Pix2pix_model_201805031054.h5'

# Data path/name
datapath = os.path.join('/','home','jmj136','deep-learning',
                        'PETrecon','CycleGAN','CycleGAN_data_TVT.hdf5')

# Training or testing
# False will only load the testing
# datset
training = True

# set learning rates and loss weights
lrD = 1e-4  # discriminator learning rate
lrG = 1e-4  # generator learning rate
λ = 10  # grad penalty weighting- leave alone
C = 200  # L1 Loss weighting

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
# True for regression (continuous) output
# False for segmentation (discrete) output
A_reg = True
B_reg = True

# whether to plot progress images
plot_prog = False
# whether to save progress gif after training
gif_prog = False
# whether to just save loss plots rather than
# displaying them
savenodisplay = False
# index of testing slice to use as progress image
ex_ind = 136
# number of iterations (batches) to train
numIter = 20000
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
LR_decay = True
LR_period = np.int(numIter/5)

#%% Loading data
# Load training, validation, and testing data
# must be in the form [slices,x,y,channels]
if not 'test_A' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        test_A = np.array(f.get('A_test'))
        test_B = np.array(f.get('B_test'))
        if training:
            train_A = np.array(f.get('A_train'))
            train_B = np.array(f.get('B_train'))
            val_A = np.array(f.get('A_val'))
            val_B = np.array(f.get('B_val'))
    print('Data Loaded')
        
#%% Setup models for training
print("Creating models...")
from keras.optimizers import Adam
import keras.backend as K

# Discriminator model
# A is input
# B is target
DisModel_B = JMJmodels.Discriminator2D(train_A.shape[1:],train_B.shape[1:],numFilters_D)

# Generator Model
GenModel_A2B = JMJmodels.CycleGANgenerator(train_A.shape[1:],train_B.shape[-1],
                                          numFilters_G,numBlocks_G,noStride,use_bn,B_reg)


# Endpoints of graph- A to B
real_A = GenModel_A2B.inputs[0]
fake_B = GenModel_A2B.outputs[0]
real_B = DisModel_B.inputs[1]

# A to B generator function
fn_genB = K.function([real_A],[fake_B])

# Discriminator scores
realBscore = DisModel_B([real_A,real_B])
fakeBscore = DisModel_B([real_A,fake_B])

# L1 score
loss_L1 = K.mean(K.abs(fake_B-real_B))
#%% B discriminator loss function
# create mixed output for gradient penalty
from keras.layers import Input
ep_input1 = K.placeholder(shape=(None,1,1,1))
mixed_B = Input(shape=train_B.shape[1:],
                    tensor=ep_input1 * real_B + (1-ep_input1) * fake_B)
mixedBscore = DisModel_B([real_A,mixed_B])
# discriminator losses
realBloss = K.mean(realBscore)
fakeBloss = K.mean(fakeBscore)
# gradient penalty loss
grad_mixed = K.gradients([mixedBscore],[mixed_B])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
grad_penalty = K.mean(K.square(norm_grad_mixed-1))
# composite Discriminator loss
loss_D = fakeBloss - realBloss + λ * grad_penalty

#%% A to B generator loss 
loss_G = -fakeBloss + C * loss_L1

#%% Training functions
# Discriminator training function
D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(DisModel_B.trainable_weights,[],loss_D)
fn_trainD = K.function([real_A, real_B, ep_input1],[loss_D], D_trups)

# Generator Training function
G_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(GenModel_A2B.trainable_weights,[], loss_G)

fn_trainG = K.function([real_A,real_B], [-fakeBloss, loss_L1], G_trups)

# validation evaluate function
fn_eval = K.function([real_A,real_B],[loss_L1])

#%% Progress plotting function
def ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax):
    A_samp = test_A[ex_ind,...][np.newaxis,...]
    [Btest] = fn_genB([A_samp])
    if B_reg:
        B_current = Btest[0,...,0]
        B_truth = test_B[ex_ind,...,0]
    else:
        Binds = np.argmax(Btest[0],axis=-1)
        B_current = Binds/Btest.shape[-1]
        Binds_truth = np.argmax(test_B[ex_ind],axis=-1)
        B_truth = Binds_truth/Btest.shape[-1]
    if A_reg:
        A_current = A_samp[0,...,0]
    else:  
        Ainds = np.argmax(A_samp[0],axis=-1)
        A_current = Ainds/test_A.shape[-1]

    samp_im = np.c_[A_current,B_current,B_truth]
    ax.imshow(samp_im,cmap='gray',vmin=0,vmax=1)
    ax.set_axis_off()
    ax.set_clip_box([0,1])
    ax.set_title('Current training state')
    plt.pause(.001)
    plt.draw()
    return samp_im



#%% training
print('Starting training...')

# preallocate for the training and validation losses
dis_loss = np.zeros((numIter,1))
gen_loss = np.zeros((numIter,2))
val_loss = np.ones((np.int(numIter/valstep),1))
templosses = np.zeros((np.int(val_A.shape[0]/val_b_s),1))

# Updatable plot
if plot_prog:
    plt.ion()
    fig, ax = plt.subplots()
    _ = ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax)
    # preallocate image display
    numProgs = np.maximum(np.int(numIter/progstep),1)
    progress_ims = np.zeros((numProgs,256,3*256))
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
        batch_inds = np.random.choice(train_A.shape[0], b_s, replace=False)
        A_batch = train_A[batch_inds,...]
        B_batch = train_B[batch_inds,...]
        # train discrimators
        ϵ1 = np.random.uniform(size=(b_s, 1, 1 ,1))
        errD  = fn_trainD([A_batch, B_batch, ϵ1])
    dis_loss[ii] = errD
    # Train Generator
    errG = fn_trainG([A_batch, B_batch])
    gen_loss[ii] = errG
    if plot_prog and ii % progstep == 0:
        # progress image plotting
        samp_im = ProgressPlot(test_A,test_B,ex_ind,A_reg,B_reg,ax)
        progress_ims[gg] = samp_im
        gg += 1
    if (ii+1) % valstep ==0:
        tqdm.write('Checking validation loss...')
        for bb in range(0,templosses.shape[0]):
            val_inds = np.arange(bb*val_b_s,np.minimum((bb+1)*val_b_s,val_A.shape[0]))
            A_batch = val_A[val_inds,...]
            valL1 = fn_eval([A_batch,B_batch])[0]
            templosses[bb] = valL1
        
        cur_val_loss = np.mean(templosses,axis=0)
        if cur_val_loss[0] <= np.min(val_loss,axis=0)[0]:
            tqdm.write('Valdiation loss decreased to {:.02e}'.format(cur_val_loss[0]))
            GenModel_A2B.save(model_filepath,True,False)
            tqdm.write("Saved models to file")
        else:
            tqdm.write('Validation loss did not decrease: {:.02e}'.format(cur_val_loss[0]))
            
        val_loss[vv] = cur_val_loss           
        vv +=1
                
    if LR_decay and (ii) % LR_period == 0 and ii > 0:
        lrG = lrG/2
        G_trups = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(GenModel_A2B.trainable_weights,[], loss_G)
        fn_trainG = K.function([real_A,real_B], [-fakeBloss, loss_L1], G_trups)
        tqdm.write('Updated generator learning rate to {:.3g}'.format(lrG))
        lrD = lrD/2
        D_trups = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(DisModel_B.trainable_weights,[],loss_D)
        fn_trainD = K.function([real_A, real_B, ep_input1],[loss_D], D_trups)
        tqdm.write('Updated discriminator learning rate to {:.3g}'.format(lrD))
        
    t.set_postfix(Dloss=dis_loss[ii,0],L1Loss = gen_loss[ii,1])
    
t.close()
del t

print('Training complete')

# backup model saving- writes as weights and network structure
# which is more reliable than saving model as single file
model_json = GenModel_A2B.to_json()
with open("BackupModel_A2B.json", "w") as json_file:
    json_file.write(model_json)
GenModel_A2B.save_weights("BackupModel_A2B.h5")

print('Backup Model saved')

# display loss
# smoothed since it's usually pretty rocky
# the L1 losses are multipled by 1000 to match scale
from scipy.signal import medfilt
fig = plt.figure()
plt.plot(np.arange(numIter),-medfilt(dis_loss[:,0],5),
         np.arange(numIter),medfilt(1000*gen_loss[:,1],5),
         np.arange(0,numIter,valstep),1000*val_loss[:])
plt.legend(['-Discriminator B Loss',
            '1000x L1 Loss',
            '1000x Validation L1 loss'])
plt.ylim([0,60])
if savenodisplay:
    name = 'pix2pixGANtrainLossPlot_{}.png'.format(timestr)
    plt.savefig(name)
else:
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
    print("Loaded backup model from weights")
else:
    # Load regular checkpointed models
    GenModel_A2B = load_model(model_filepath.format('A2B'),None,False)

# get generator results
time1 = time.time()
Btest = GenModel_A2B.predict(test_A)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/test_A.shape[0],' ms per slice')

# Display progress images, if created
if plot_prog:
    output_file = 'ProgressIms_{}.gif'.format(timestr)
    multi_slice_viewer0(progress_ims,'Training Progress Images')
    
    # save progress ims to gif
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

if B_reg:
    B_output = Btest[...,0]
    B_truth = test_B[...,0]
else:
    Binds = np.argmax(Btest,axis=-1)
    B_output = Binds/Btest.shape[-1]
    Binds_truth = np.argmax(test_B,axis=-1)
    B_truth = Binds_truth/Btest.shape[-1]
if A_reg:
    A_output = test_A[...,0]
else:
    Ainds = np.argmax(test_A,axis=-1)
    A_output = Ainds/test_A.shape[-1]
        
test_disp = np.c_[A_output,B_output,B_truth]
if B_reg:
    multi_slice_viewer0(test_disp,'Test Images',SSIMs)
else:
    multi_slice_viewer0(test_disp,'Test Images')

# Load full A data and run inferencing for visual inspection
#with h5py.File('CycleGAN_FullAdata.hdf5','r') as f:
#        full_A = np.array(f.get('A_full'))
#full_B = GenModel_A2B.predict(full_A)
#full_Arec = GenModel_B2A.predict(full_B)
#full_disp = np.concatenate((full_A[...,0],full_B[...,0],full_Arec[...,0]),axis=2)
#multi_slice_viewer0(full_disp,'Full Images')
