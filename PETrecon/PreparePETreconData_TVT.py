# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import glob
import scipy.io as spio
import skimage.exposure as skexp
from skimage.transform import rescale
import os

DataDir = 'MatData'
filepath = "{}/PETreconTrainingData{:03d}.mat".format(DataDir, 1)
datapath = 'petrecondata_tvt.hdf5'
numsubj = 11
val_split = .2
test_split = .2
np.random.seed(seed=1)
#%% Inputs
print('Loading inputs')
globfiles = sorted(glob.glob("MatData/*.mat"))
mat = spio.loadmat(globfiles[0],squeeze_me=True)
ims = mat['x']
ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
wims = ims[...,0]
fims = ims[...,1]
for im in wims:
    im /= np.max(im)
for im in fims:
    im /= np.max(im)

inputs = np.stack((wims,fims),axis=3)

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    ims = mat['x']
    ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
    wims = ims[...,0]
    fims = ims[...,1]
    for im in wims:
        im /= np.max(im)
    for im in fims:
        im /= np.max(im)
    
    new_inputs = np.stack((wims,fims),axis=3)
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    
inputs[inputs<0] = 0

#%% Targets
print('Loading targets')
mat = spio.loadmat(globfiles[0],squeeze_me=True)
ims = mat['y']
ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
targets = ims[...,np.newaxis]

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    ims = mat['y']
    ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
    new_targets = ims[...,np.newaxis]
    targets = np.concatenate((targets,new_targets),axis=0)
    
#%% Split validation,testing data
numSamples = inputs.shape[0]
val_inds=np.random.choice(np.arange(numSamples),np.round(val_split*numSamples).astype(np.int),replace=False)

val_inputs = np.take(inputs,val_inds,axis=0)
val_targets = np.take(targets,val_inds,axis=0)

inputs = np.delete(inputs,val_inds,axis=0)
targets = np.delete(targets,val_inds,axis=0)

test_inds=np.random.choice(np.arange(inputs.shape[0]),np.round(test_split*numSamples).astype(np.int),replace=False)

test_inputs = np.take(inputs,test_inds,axis=0)
test_targets = np.take(targets,test_inds,axis=0)

inputs = np.delete(inputs,test_inds,axis=0)
targets = np.delete(targets,test_inds,axis=0)

# store validation and testing data to HDF5 file
print('storing validation and testing data as HDF5')
try:
    with h5py.File(datapath, 'x') as hf:
        hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
        hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
    with h5py.File(datapath, 'a') as hf:
        hf.create_dataset("val_targets",  data=val_targets,dtype='f')
        hf.create_dataset("test_targets",  data=test_targets,dtype='f')
except Exception as e:
    os.remove(datapath)
    with h5py.File(datapath, 'x') as hf:
        hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
        hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
    with h5py.File(datapath, 'a') as hf:
        hf.create_dataset("val_targets",  data=val_targets,dtype='f')
        hf.create_dataset("test_targets",  data=test_targets,dtype='f')

#%% augment training data

# LR flips
fl_inputs = np.flip(inputs,2)
fl_targets = np.flip(targets,2)

# gamma corrections
gammas = .5 + np.random.rand(inputs.shape[0])
gm_inputs = np.copy(inputs)
for ii in range(gm_inputs.shape[0]):
    gm_inputs[ii,...,0] = skexp.adjust_gamma(gm_inputs[ii,...,0],gamma=gammas[ii])
    gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    
gm_targets = np.copy(targets)

# combine together
aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)
aug_targets = np.concatenate((targets,fl_targets,gm_targets),axis=0)


#%% finalize training data

# randomize inputs
print('randomizing training inputs')
numS = aug_inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(aug_inputs,sort_r,axis=0,out=aug_inputs)
np.take(aug_targets,sort_r,axis=0,out=aug_targets)


# store training data
print('storing train data as HDF5')
with h5py.File(datapath, 'a') as hf:
    hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
with h5py.File(datapath, 'a') as hf:
    hf.create_dataset("train_targets",  data=aug_targets,dtype='f')
    
print('done')

del val_inputs
del val_targets
del test_inputs
del test_targets
del aug_inputs
del aug_targets
del inputs
del fl_inputs
del gm_inputs
del targets
del fl_targets
del gm_targets
