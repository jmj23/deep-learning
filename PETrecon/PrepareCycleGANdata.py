# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import ants

datapath = 'CycleRegNIFTIs/subj{:03d}_{}.nii'
savepath = 'DeepMuMapCycleGAN_data.hdf5'

subj_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
val_frac = .2

np.random.seed(seed=1)

eps = 1e-12

#%% Inputs
print('Loading inputs')

subj = subj_vec[0]
wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
for im in wims:
    im[im<0]=0
    im /= np.max(im)
for im in fims:
    im[im<0]=0
    im /= np.max(im)
for im in inims:
    im[im<0]=0
    im /= np.max(im)
for im in outims:
    im[im<0]=0
    im /= (np.max(im)+eps)
inputs = np.stack((wims,fims,inims,outims),axis=3)

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
    for im in wims:
        im[im<0]=0
        im /= np.max(im)
    for im in fims:
        im[im<0]=0
        im /= np.max(im)
    for im in inims:
        im[im<0]=0
        im /= np.max(im)
    for im in outims:
        im[im<0]=0
        im /= (np.max(im)+eps)
    inputarray = np.stack((wims,fims,inims,outims),axis=3)
    inputs = np.concatenate((inputs,inputarray),axis=0)
    
inputs = np.rot90(inputs,k=-1,axes=(1,2))
    
#%% HU-LAC conversion
def ConvertToLAC(CTims):
    muMap = np.copy(CTims)
    # new conversion
    muMap[CTims<=30] = 9.6e-5*(CTims[CTims<=30]+1024)
    muMap[CTims>30]= 5.64e-5*(CTims[CTims>30]+1024)+4.08e-2
    muMap[muMap<0] = 0
    return muMap*5

#%% Remove empty CT slices
def ClipEmptySlices(CTims):
    ims = CTims - np.min(CTims)
    sums = np.sum(ims>1,axis=(1,2))
    good_inds = sums>11e3
    return CTims[good_inds]

#%%
print('Loading targets')

subj = subj_vec[0]
CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
CTims = ClipEmptySlices(CTims)

# convert HU to LAC
muMap = ConvertToLAC(CTims)

targets = muMap[...,np.newaxis]

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
    CTims = ClipEmptySlices(CTims)
    muMap = ConvertToLAC(CTims)
    new_targets = muMap[...,np.newaxis]
    targets = np.concatenate((targets,new_targets),axis=0)

targets = np.rot90(targets,k=-1,axes=(1,2))
#%% Split validation from training data
# Get validation data
val_inds = np.random.choice(inputs.shape[0], np.int(inputs.shape[0]*val_frac), replace=False)
val_inputs = np.take(inputs,val_inds,axis=0)
# Remove from training arrays
inputs = np.delete(inputs,val_inds,axis=0)

#%% store validation and testing data to HDF5 file
print('Storing validation and testing data as HDF5...')
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset("MR_train",  data=inputs,dtype='f')
    hf.create_dataset("MR_val", data=val_inputs,dtype='f')
    hf.create_dataset("CT_train",  data=targets,dtype='f')

print('done')
#%%
del val_inputs
del inputs
del inputarray
del targets