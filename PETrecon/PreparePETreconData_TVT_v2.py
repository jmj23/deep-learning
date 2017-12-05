# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import skimage.exposure as skexp
import os
import ants

datapath = 'RegNIFTIs/subj{:03d}_{}.nii'
savepath = 'petrecondata_tvt_v2.hdf5'

subj_vec = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
val_num = 3
test_num = 2
np.random.seed(seed=1)

#%% Inputs
print('Loading inputs')

subj = subj_vec[0]
wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
for im in wims:
    im /= np.max(im)
for im in fims:
    im /= np.max(im)
for im in inims:
    im /= np.max(im)
inputarray = np.stack((wims,fims,inims),axis=3)
inputs = inputarray[good_inds]
sliceNums = [inputs.shape[0]]

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    for im in wims:
        im /= np.max(im)
    for im in fims:
        im /= np.max(im)
    for im in inims:
        im /= np.max(im)
    inputarray = np.stack((wims,fims,inims),axis=3)
    new_inputs = inputarray[good_inds]
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    sliceNums.append(new_inputs.shape[0])
       

#%% Targets
print('Loading targets')

subj = subj_vec[0]
CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)

# convert HU to LAC
muMap = np.copy(CTims)
muMap[CTims<=0] = (1+CTims[CTims<=0]/1000)*0.096
muMap[CTims>0] = (1+CTims[CTims>0]*6.4*10e-4)*0.096
muMap[muMap<0] = 0
muMap *= 5


targets = muMap[good_inds][...,np.newaxis]

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
    good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    muMap = np.copy(CTims)
    muMap[CTims<=0] = (1+CTims[CTims<=0]/1000)*0.096
    muMap[CTims>0] = (1+CTims[CTims>0]*6.4*10e-4)*0.096
    muMap[muMap<0] = 0
    muMap *=5
    new_targets = muMap[good_inds][...,np.newaxis]
    targets = np.concatenate((targets,new_targets),axis=0)
    
#%% Split validation,testing data
# get slice ranges corresponding to subjects
sliceNumArray = np.array(sliceNums)
sliceCumArray = np.cumsum(sliceNumArray)
# pick out subjects
numSubjs = sliceNumArray.shape[0]
sep_subjs = np.random.choice(np.arange(numSubjs),val_num+test_num,replace=False)
val_subjs = sep_subjs[:val_num]
test_subjs = sep_subjs[val_num:]
# Get validation data
val_inds = np.concatenate(([np.arange(sliceCumArray[ind],sliceCumArray[ind+1]) for ind in val_subjs]))
val_inputs = np.take(inputs,val_inds,axis=0)
val_targets = np.take(targets,val_inds,axis=0)
# Get testing data
test_inds = np.concatenate(([np.arange(sliceCumArray[ind],sliceCumArray[ind+1]) for ind in test_subjs]))
test_inputs = np.take(inputs,test_inds,axis=0)
test_targets = np.take(targets,test_inds,axis=0)
# Remove from training arrays
remove_inds = np.concatenate((test_inds,val_inds))
inputs = np.delete(inputs,remove_inds,axis=0)
targets = np.delete(targets,remove_inds,axis=0)


# store validation and testing data to HDF5 file
print('Storing validation and testing data as HDF5...')
try:
    with h5py.File(savepath, 'x') as hf:
        hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
        hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
    with h5py.File(savepath, 'a') as hf:
        hf.create_dataset("val_targets",  data=val_targets,dtype='f')
        hf.create_dataset("test_targets",  data=test_targets,dtype='f')
except Exception as e:
    os.remove(savepath)
    with h5py.File(savepath, 'x') as hf:
        hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
        hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
    with h5py.File(savepath, 'a') as hf:
        hf.create_dataset("val_targets",  data=val_targets,dtype='f')
        hf.create_dataset("test_targets",  data=test_targets,dtype='f')

#%% augment training data

# LR flips
fl_inputs = np.flip(inputs,1)
fl_targets = np.flip(targets,1)

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
print('Randomizing training inputs...')
numS = aug_inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(aug_inputs,sort_r,axis=0,out=aug_inputs)
np.take(aug_targets,sort_r,axis=0,out=aug_targets)


# store training data
print('Storing train data as HDF5...')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("train_targets",  data=aug_targets,dtype='f')
    
print('done')
#%%
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
