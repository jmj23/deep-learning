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

DataDir = 'Data'
filepath = "{}/dataset{}.mat".format(DataDir, 1)

#%% Inputs
mat = spio.loadmat(filepath,squeeze_me=True)
inputs = mat['InputArray']
inputs = inputs.astype(np.float32, copy=False)

globfiles = glob.glob("Data/*.mat")

for file in globfiles[1:]:
    print(file[-7:])
    mat = spio.loadmat(file,squeeze_me=True)
    inputs = np.concatenate((inputs,mat['InputArray']),axis=0)
    
# synthesize more data
# LR flips
fl_inputs = np.flip(inputs,2)

# combine together
aug_inputs = np.concatenate((inputs,fl_inputs),axis=0)

# pull out validation set
val_rat = 0.15
numS = aug_inputs.shape[0]
numVal = np.round(val_rat*numS).astype(np.int)
valvec = np.random.choice(numS, numVal, replace=False)

val_inputs = np.take(aug_inputs,valvec,axis=0)
inputs = np.delete(aug_inputs,valvec,axis=0)
    
# randomize inputs
print('sorting')
numS = inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(inputs,sort_r,axis=0,out=inputs)

# store to HDF5 file
print('storing as HDF5')
with h5py.File('segdata_train_aug.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
del inputs
with h5py.File('segdata_val_aug.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=val_inputs,dtype='f')
del val_inputs
del aug_inputs
#%% Targets  
mat = spio.loadmat(filepath,squeeze_me=True)
targets = mat['TargetArray'][:,:,:,0]
targets = np.expand_dims(targets, 3)
targets = targets.astype(np.int32, copy=False)

for file in globfiles[1:]:
    print(file[-7:])
    mat = spio.loadmat(file,squeeze_me=True)
    addtl = mat['TargetArray'][:,:,:,0]
    addtl = np.expand_dims(addtl, 3)
    targets = np.concatenate((targets,addtl),axis=0)
    
    
# synthesize extra data
# LR flips
fl_targets = np.flip(targets,2)

# combine together
aug_targets = np.concatenate((targets,fl_targets),axis=0)

# pull out validation set
val_targets = np.take(aug_targets,valvec,axis=0)
targets = np.delete(aug_targets,valvec,axis=0)
    
# sort targets same as inputs
print('sorting')
np.take(targets,sort_r,axis=0,out=targets)

print('saving to hdf5')
with h5py.File('segdata_train_aug.hdf5', 'a') as hf:
    hf.create_dataset("targets",  data=targets,dtype='i')
with h5py.File('segdata_val_aug.hdf5', 'a') as hf:
    hf.create_dataset("targets", data=val_targets,dtype='i')
    
print('done')

del targets
del val_targets
del aug_targets