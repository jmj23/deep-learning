# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import glob
import scipy.io as spio

DataDir = 'MatData'
filepath = "{}/segdata{:03d}.mat".format(DataDir, 1)
numsubj = 61
#%% Inputs
globfiles = sorted(glob.glob("MatData/*.mat"))
mat = spio.loadmat(globfiles[0],squeeze_me=True)
waterims = mat['imagesW']
waterims = np.rollaxis(waterims.astype(np.float32, copy=False),2,0)
fatims = mat['imagesF']
fatims = np.rollaxis(fatims.astype(np.float32, copy=False),2,0)
Vcut = mat['Vmask']
Vcut = np.rollaxis(Vcut.astype(np.float32, copy=False),2,0)
topind = mat['topind']
botind = mat['botind']
for im in waterims:
    im /= np.max(im)
for im in fatims:
    im /= np.max(im)

numS = topind-botind
waterims = waterims[botind:topind]
fatims = fatims[botind:topind]
Vcut = Vcut[botind:topind]
inputs = np.stack((waterims,fatims,Vcut),axis=-1)

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    waterims = mat['imagesW']
    waterims = np.rollaxis(waterims.astype(np.float32, copy=False),2,0)
    fatims = mat['imagesF']
    fatims = np.rollaxis(fatims.astype(np.float32, copy=False),2,0)
    Vcut = mat['Vmask']
    Vcut = np.rollaxis(Vcut.astype(np.float32, copy=False),2,0)
    topind = mat['topind']
    botind = mat['botind']
    for im in waterims:
        im /= np.max(im)
    for im in fatims:
        im /= np.max(im)
    
    numS = topind-botind
    waterims = waterims[botind:topind]
    fatims = fatims[botind:topind]
    Vcut = Vcut[botind:topind]
    new_inputs = np.stack((waterims,fatims,Vcut),axis=-1)
    inputs = np.concatenate((inputs,new_inputs),axis=0)

# store to HDF5 file
print('storing as HDF5')
with h5py.File('batch_segdata.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
del inputs

#%% Targets  
mat = spio.loadmat(globfiles[0],squeeze_me=True)
topind = mat['topind']
botind = mat['botind']
numS = topind-botind
SegMask = mat['SegMask']
SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
targets = SegMask[botind:topind].reshape(numS,256,256,1)

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    topind = mat['topind']
    botind = mat['botind']
    numS = topind-botind
    SegMask = mat['SegMask']
    SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
    new_targets = SegMask[botind:topind].reshape(numS,256,256,1)
    targets = np.concatenate((targets,new_targets),axis=0)
    
print('saving to hdf5')
with h5py.File('batch_segdata.hdf5', 'a') as hf:
    hf.create_dataset("targets",  data=targets,dtype='i')
    
print('done')

del targets