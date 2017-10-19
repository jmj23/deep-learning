# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import glob
import scipy.io as spio

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
    
# sort inputs
# determine water signal
print('determining sort order')
sort_vals = np.sum(inputs[...,0]>.1,axis=(1,2))
sort_r = np.argsort(-sort_vals)
print('sorting')
np.take(inputs,sort_r,axis=0,out=inputs)

# store to HDF5 file
print('storing as HDF5')
with h5py.File('segdata_sort.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
del inputs

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
    
# sort targets same as inputs
print('sorting')
np.take(targets,sort_r,axis=0,out=targets)

print('saving to hdf5')
with h5py.File('segdata_sort.hdf5', 'a') as hf:
    hf.create_dataset("targets",  data=targets,dtype='i')
print('done')

del targets