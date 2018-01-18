# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import skimage.exposure as skexp
import nibabel as nib
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
savepath = 'lowdosePETdata_v2.hdf5'

train_subj_vec = [3,5,7,8,9,10,11,12,13,15,17]
val_subj_vec = [2, 16]
test_subj_vec = [14, 4, 6]
np.random.seed(seed=2)
eps = 1e-8

normfac = 20000 # what the images are normalized to. Keep this is mind when
                # looking at images afterwards

#%% Training data loading functions
# Inputs
def load_training_input(subj):
    waterpath = 'RegNIFTIs/subj{:03d}_WATER.nii'.format(subj)
    fatpath = 'RegNIFTIs/subj{:03d}_FAT.nii'.format(subj)
    LDpath = 'lowdose/volunteer{:03d}_lowdose.nii.gz'.format(subj)
    LDims = nib.load(LDpath).get_data()
    LDims = np.rollaxis(np.rot90(np.rollaxis(LDims,2,0),1,axes=(1,2)),3,0)
    frame1 = LDims[0]
    for im in frame1:
        im[im<0] = 0
        im /= normfac
    wims = np.rot90(np.rollaxis(nib.load(waterpath.format(subj)).get_data(),2,0),1,axes=(1,2))
    for im in wims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    fims = np.rot90(np.rollaxis(nib.load(fatpath.format(subj)).get_data(),2,0),1,axes=(1,2))
    for im in fims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    inputs = np.stack((frame1,wims,fims),axis=3)
    
    for fnum in range(1,5):
        frame = LDims[fnum]
        for im in frame:
            im[im<0]=0
            im /= normfac
        wims = np.rot90(np.rollaxis(nib.load(waterpath.format(subj)).get_data(),2,0),1,axes=(1,2))
        for im in wims:
            im[im<0] = 0
            im /= (np.max(im)+eps)
        fims = np.rot90(np.rollaxis(nib.load(fatpath.format(subj)).get_data(),2,0),1,axes=(1,2))
        for im in fims:
            im[im<0] = 0
            im /= (np.max(im)+eps)
        inputarray = np.stack((frame,wims,fims),axis=3)
        inputs = np.concatenate((inputs,inputarray),axis=0)
    return inputs

# Target
def load_training_target(subj):
    FDpath = 'fulldose/volunteer{:03d}_fulldose.nii.gz'.format(subj)
    FDims = nib.load(FDpath).get_data()
    FDims = np.rot90(np.rollaxis(FDims,2,0),1,axes=(1,2))
    for im in FDims:
        im[im<0]=0
        im /= normfac
    targets = np.tile(FDims[...,np.newaxis],(5,1,1,1))
    return targets

#%% Loading training inputs
print('Loading inputs')
subj = train_subj_vec[0]
print('Loading training subject',subj,'...')
inputs = load_training_input(subj)

for subj in train_subj_vec[1:]:
    print('Loading training subject',subj,'...')
    newinputs = load_training_input(subj)
    inputs = np.concatenate((inputs,newinputs),axis=0)
    
#%% Load training targets
print('Loading training targets')
subj = train_subj_vec[0]
print('Loading training subject',subj,'...')
targets = load_training_target(subj)

for subj in train_subj_vec[1:]:
    print('Loading training subject',subj,'...')
    newtargets = load_training_target(subj)
    targets = np.concatenate((targets,newtargets),axis=0)
    
#%% augment training data
print('Augmenting training data...')
# LR flips
fl_inputs = np.flip(inputs,2)
fl_targets = np.flip(targets,2)

# gamma corrections
gammas = .5 + np.random.rand(inputs.shape[0])
gm_inputs = np.copy(inputs)
for ii in range(gm_inputs.shape[0]):
    gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    gm_inputs[ii,...,2] = skexp.adjust_gamma(gm_inputs[ii,...,2],gamma=gammas[ii])
    
gm_targets = np.copy(targets)

# combine together
aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)
aug_targets = np.concatenate((targets,fl_targets,gm_targets),axis=0)


#%% finalize training data
# randomize inputs
print('Randomizing training inputs...')
numS = aug_inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(aug_inputs,sort_r,axis=0,out=aug_inputs)
np.take(aug_targets,sort_r,axis=0,out=aug_targets)


# store training data
print('Storing train data as HDF5...')
with h5py.File(savepath, 'x') as hf:
    hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("train_targets",  data=aug_targets,dtype='f')
    
#%%
del aug_inputs
del aug_targets
del inputs
del fl_inputs
del gm_inputs
del targets
del fl_targets
del gm_targets
del newinputs
del newtargets

#%% Validation data
def load_eval_input(subj,frame=1):
    print('Loading evaluation subject',subj,'...')
    waterpath = 'RegNIFTIs/subj{:03d}_WATER.nii'.format(subj)
    fatpath = 'RegNIFTIs/subj{:03d}_FAT.nii'.format(subj)
    LDpath = 'lowdose/volunteer{:03d}_lowdose.nii.gz'.format(subj)
    LDims = nib.load(LDpath).get_data()
    LDims = np.rollaxis(np.rot90(np.rollaxis(LDims,2,0),1,axes=(1,2)),3,0)
    frame1 = LDims[0]
    for im in frame1:
        im[im<0] = 0
        im /= normfac
    wims = np.rot90(np.rollaxis(nib.load(waterpath.format(subj)).get_data(),2,0),-1,axes=(1,2))
    for im in wims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    fims = np.rot90(np.rollaxis(nib.load(fatpath.format(subj)).get_data(),2,0),-1,axes=(1,2))
    for im in fims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    inputs = np.stack((frame1,wims,fims),axis=3)
    return inputs

def load_eval_target(subj):
    FDpath = 'fulldose/volunteer{:03d}_fulldose.nii.gz'.format(subj)
    FDims = nib.load(FDpath).get_data()
    FDims = np.rot90(np.rollaxis(FDims,2,0),1,axes=(1,2))
    for im in FDims:
        im[im<0]=0
        im /= normfac
    targets = FDims[...,np.newaxis]
    return targets

val_inputs = load_eval_input(val_subj_vec[0],1)
newinput = load_eval_input(val_subj_vec[1],1)
val_inputs = np.concatenate((val_inputs,newinput),axis=0)
val_targets = load_eval_target(val_subj_vec[0])
newtarget = load_eval_target(val_subj_vec[1])
val_targets = np.concatenate((val_targets,newtarget),axis=0)

# Testing data
test_inputs = load_eval_input(test_subj_vec[0],1)
newinput = load_eval_input(test_subj_vec[1],1)
test_inputs = np.concatenate((test_inputs,newinput),axis=0)
newinput = load_eval_input(test_subj_vec[2],1)
test_inputs = np.concatenate((test_inputs,newinput),axis=0)
test_targets = load_eval_target(test_subj_vec[0])
newtarget = load_eval_target(test_subj_vec[1])
test_targets = np.concatenate((test_targets,newtarget),axis=0)
newtarget = load_eval_target(test_subj_vec[2])
test_targets = np.concatenate((test_targets,newtarget),axis=0)
    
    
#%% Storing validation and testing data
print('Storing validation and testing data as HDF5...')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
    hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("val_targets",  data=val_targets,dtype='f')
    hf.create_dataset("test_targets",  data=test_targets,dtype='f')
#%%
del val_inputs
del val_targets
del test_inputs
del test_targets

print('done')