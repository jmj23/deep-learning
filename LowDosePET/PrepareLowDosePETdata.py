# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
#import skimage.exposure as skexp
import nibabel as nib
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

multiSlice = 3

LDtype = '30s'
#LDtype = '60s'

if LDtype == '30s':
    savepath = 'lowdosePETdata_30s.hdf5'
elif LDtype == '60s':
    savepath = 'lowdosePETdata_60s.hdf5'
    
if LDtype == '30s':
    LowDosePath = 'lowdose_30s/volunteer{:03d}_lowdose.nii.gz'
elif LDtype == '60s':
    LowDosePath = 'lowdose_60s/volunteer{:03d}_lowdose.nii.gz'

train_subj_vec = [3,5,7,8,9,10,11,12,13,15,17]
val_subj_vec = [2, 16]
test_subj_vec = [14, 4, 6]
np.random.seed(seed=2)
eps = 1e-8
numFrames = 3
sCO = 3 # slice cutoff
normfac = 20000 # what the images are normalized to. Keep this is mind when
                # looking at images afterwards

#%% Training data loading functions
def ConvertToMultiSlice(array,MS=3):
    tshp = array.shape
    MSos = np.int((MS-1)/2) #MultiSlice Offset
    MSarray = np.zeros((tshp[0],MS,tshp[1],tshp[2],tshp[3]))
    for ss in range(MSos,tshp[0]-MSos):
        MSarray[ss,0,...] = array[ss-1]
        MSarray[ss,1,...] = array[ss]
        MSarray[ss,2,...] = array[ss+1]
    MSarray[0,0,...] = array[0]
    MSarray[0,1,...] = array[0]
    MSarray[0,2,...] = array[1]
    MSarray[-1,0,...] = array[-2]
    MSarray[-1,1,...] = array[-1]
    MSarray[-1,2,...] = array[-1]
    return MSarray
# Inputs
def load_training_input(subj):
    waterpath = 'RegNIFTIs/subj{:03d}_WATER.nii'.format(subj)
    fatpath = 'RegNIFTIs/subj{:03d}_FAT.nii'.format(subj)
    LDpath = LowDosePath.format(subj)
    LDims = nib.load(LDpath).get_data()
    LDims = np.rollaxis(np.rot90(np.rollaxis(LDims,2,0),1,axes=(1,2)),3,0)
    frame1 = LDims[0]
    for im in frame1:
        im[im<0] = 0
        im /= normfac
    wnft = nib.as_closest_canonical(nib.load(waterpath.format(subj)))
    wims = np.flip(np.rot90(np.rollaxis(wnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    for im in wims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    
    fnft = nib.as_closest_canonical(nib.load(fatpath.format(subj)))
    fims= np.flip(np.rot90(np.rollaxis(fnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    for im in fims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
    inputs = np.stack((frame1,wims,fims),axis=3)[sCO:-sCO,...]
    inputs = ConvertToMultiSlice(inputs,multiSlice)
    
    for fnum in range(1,numFrames):
        frame = LDims[fnum]
        for im in frame:
            im[im<0]=0
            im /= normfac
            
        inputarray = np.stack((frame,wims,fims),axis=3)[sCO:-sCO,...]
        inputarray = ConvertToMultiSlice(inputarray,multiSlice)
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
    targets = np.tile(FDims[sCO:-sCO,...,np.newaxis],(numFrames,1,1,1))
    return targets

#%% Load, augment and save training inputs
print('Loading inputs')
subj = train_subj_vec[0]
print('Loading training subject',subj,'...')
inputs = load_training_input(subj)

for subj in train_subj_vec[1:]:
    print('Loading training subject',subj,'...')
    newinputs = load_training_input(subj)
    inputs = np.concatenate((inputs,newinputs),axis=0)

## Augment training inputs
#print('Augmenting training inputs...')
## LR flips
#fl_inputs = np.flip(inputs,3)
## gamma corrections
#gammas = .5 + np.random.rand(inputs.shape[0],inputs.shape[-1])
#gm_inputs = np.copy(inputs)
#for ii in range(gm_inputs.shape[0]):
#    for jj in range(gm_inputs.shape[-1]):
#        gm_inputs[ii,...,jj] = skexp.adjust_gamma(gm_inputs[ii,...,jj],gamma=gammas[ii,jj])
#        
## combine together
#aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)
aug_inputs = inputs

# store training data
print('Storing training inputs as HDF5...')
try:
    with h5py.File(savepath, 'x') as hf:
        hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
except Exception as e:
    os.remove(savepath)
    with h5py.File(savepath, 'x') as hf:
        hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
        
del aug_inputs
del inputs
#del fl_inputs
#del gm_inputs
del newinputs

#%% Load, augment, and save training targets
print('Loading training targets')
subj = train_subj_vec[0]
print('Loading training subject',subj,'...')
targets = load_training_target(subj)

for subj in train_subj_vec[1:]:
    print('Loading training subject',subj,'...')
    newtargets = load_training_target(subj)
    targets = np.concatenate((targets,newtargets),axis=0)
    
## augment training targets
#print('Augmenting training targets...')
## LR flips
#fl_targets = np.flip(targets,3)
## gamma corrections    
#gm_targets = np.copy(targets)
#
## combine together
#aug_targets = np.concatenate((targets,fl_targets,gm_targets),axis=0)
aug_targets = targets

# store training data
print('Storing training targets as HDF5...')
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("train_targets",  data=aug_targets,dtype='f')

#%%
del targets
#del fl_targets
#del gm_targets
del newtargets

#%% Validation data
def load_eval_input(subj,frame=1):
    print('Loading evaluation input for subject',subj,'...')
    waterpath = 'RegNIFTIs/subj{:03d}_WATER.nii'.format(subj)
    fatpath = 'RegNIFTIs/subj{:03d}_FAT.nii'.format(subj)
    LDpath = LowDosePath.format(subj)
    LDims = nib.load(LDpath).get_data()
    LDims = np.rollaxis(np.rot90(np.rollaxis(LDims,2,0),1,axes=(1,2)),3,0)
    frame1 = LDims[0]
    for im in frame1:
        im[im<0] = 0
        im /= normfac
        
    wnft = nib.as_closest_canonical(nib.load(waterpath.format(subj)))
    wims = np.flip(np.rot90(np.rollaxis(wnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    for im in wims:
        im[im<0] = 0
        im /= (np.max(im)+eps)

    fnft = nib.as_closest_canonical(nib.load(fatpath.format(subj)))
    fims= np.flip(np.rot90(np.rollaxis(fnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    for im in fims:
        im[im<0] = 0
        im /= (np.max(im)+eps)
        
    inputs = np.stack((frame1,wims,fims),axis=3)[sCO:-sCO,...]
    return ConvertToMultiSlice(inputs,multiSlice)

def load_eval_target(subj):
    print('Loading evaluation target for subject',subj,'...')
    FDpath = 'fulldose/volunteer{:03d}_fulldose.nii.gz'.format(subj)
    FDims = nib.load(FDpath).get_data()
    FDims = np.rot90(np.rollaxis(FDims,2,0),1,axes=(1,2))
    for im in FDims:
        im[im<0]=0
        im /= normfac
    targets = FDims[sCO:-sCO,...,np.newaxis]
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
#from VisTools import multi_slice_viewer0
#dnum = 500
#disp_inds = np.random.choice(aug_inputs.shape[0], dnum, replace=False)
#multi_slice_viewer0(np.c_[aug_inputs[disp_inds,...,0],aug_inputs[disp_inds,...,1],aug_targets[disp_inds,...,0]],'Training data')
#multi_slice_viewer0(np.c_[val_inputs[...,0],val_inputs[...,1],val_targets[...,0]],'Validation Data')
#multi_slice_viewer0(np.c_[test_inputs[...,0],test_inputs[...,1],test_targets[...,0]],'Test data')

del val_inputs
del val_targets
del test_inputs
del test_targets

print('done')