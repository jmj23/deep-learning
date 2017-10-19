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

DataDir = 'MatData'
filepath = "{}/segdata{:03d}.mat".format(DataDir, 1)
numsubj = 120
val_split = .2
np.random.seed(seed=1)
#%% Inputs
print('Loading inputs')
globfiles = sorted(glob.glob("MatData/*.mat"))
mat = spio.loadmat(globfiles[0],squeeze_me=True)
waterims = mat['imagesW']
waterims = np.rollaxis(waterims.astype(np.float32, copy=False),2,0)
fatims = mat['imagesF']
fatims = np.rollaxis(fatims.astype(np.float32, copy=False),2,0)
Vcut = mat['Vmask']
Vcut = np.rollaxis(Vcut.astype(np.float32, copy=False),2,0)
for im in waterims:
    im /= np.max(im)
for im in fatims:
    im /= np.max(im)
if waterims.shape[0] != 40:
    waterims = waterims[2:42]
    fatims = fatims[2:42]
    Vcut = Vcut[2:42]
inputs = np.stack((waterims,fatims,Vcut),axis=-1)[np.newaxis,...]

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    waterims = mat['imagesW']
    waterims = np.rollaxis(waterims.astype(np.float32, copy=False),2,0)
    fatims = mat['imagesF']
    fatims = np.rollaxis(fatims.astype(np.float32, copy=False),2,0)
    Vcut = mat['Vmask']
    Vcut = np.rollaxis(Vcut.astype(np.float32, copy=False),2,0)
    for im in waterims:
        im /= np.max(im)
    for im in fatims:
        im /= np.max(im)
    
    if waterims.shape[0] != 40:
        waterims = waterims[2:42]
        fatims = fatims[2:42]
        Vcut = Vcut[2:42]
    new_inputs = np.stack((waterims,fatims,Vcut),axis=-1)[np.newaxis,...]
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    
#%% Targets
print('Loading targets')
mat = spio.loadmat(globfiles[0],squeeze_me=True)
SegMask = mat['SegMask']
SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
if SegMask.shape[0] != 40:
        SegMask = SegMask[2:42]
targets = SegMask.reshape(1,40,256,256,1)

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    SegMask = mat['SegMask']
    SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
    if SegMask.shape[0] != 40:
        SegMask = SegMask[2:42]
    new_targets = SegMask.reshape(1,40,256,256,1)
    targets = np.concatenate((targets,new_targets),axis=0)
    
    
#%% pull out validation set
val_rat = 0.15
numS = inputs.shape[0]
numVal = np.round(val_rat*numS).astype(np.int)
valvec = np.random.choice(numS, numVal, replace=False)

val_inputs = np.take(inputs,valvec,axis=0)
inputs = np.delete(inputs,valvec,axis=0)

val_targets = np.take(targets,valvec,axis=0)
targets = np.delete(targets,valvec,axis=0)

print('storing validation data as HDF5')
with h5py.File('segdata_3D_val.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=val_inputs,dtype='f')
    hf.create_dataset("targets", data=val_targets,dtype='i')
del val_inputs
del val_targets

#%% augment
print('Augmenting data')
fl_inputs = np.flip(inputs,3)
fl_targets = np.flip(targets,3)

# gamma corrections
gammas = .5 + np.random.rand(inputs.shape[0])
gm_inputs = np.copy(inputs)
for ii in range(gm_inputs.shape[0]):
    gm_inputs[ii,...,0] = skexp.adjust_gamma(gm_inputs[ii,...,0],gamma=gammas[ii])
    gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    
gm_targets = np.copy(targets)

# combine together
aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)
del inputs, fl_inputs, gm_inputs
aug_targets = np.concatenate((targets,fl_targets,gm_targets),axis=0)
del targets, fl_targets, gm_targets

# store to HDF5 file
print('storing training data as HDF5')
with h5py.File('segdata_3D_train.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=aug_inputs,dtype='f')
    hf.create_dataset("targets",  data=aug_targets,dtype='i')
    
del aug_inputs
del aug_targets

print('Done')