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

numS = topind-botind+1
waterims = waterims[botind:topind+1]
fatims = fatims[botind:topind+1]
Vcut = Vcut[botind:topind+1]
inputs = np.stack((waterims,fatims,Vcut),axis=-1)

for file in globfiles[1:]:
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
    
    numS = topind-botind+1
    waterims = waterims[botind:topind+1]
    fatims = fatims[botind:topind+1]
    Vcut = Vcut[botind:topind+1]
    new_inputs = np.stack((waterims,fatims,Vcut),axis=-1)
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    
# synthesize more data

# LR flips
fl_inputs = np.flip(inputs,2)

# gamma corrections
gammas = .5 + np.random.rand(inputs.shape[0])
gm_inputs = np.copy(inputs)
for ii in range(gm_inputs.shape[0]):
    gm_inputs[ii,...,0] = skexp.adjust_gamma(gm_inputs[ii,...,0],gamma=gammas[ii])
    gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    
# combine together
aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)


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
with h5py.File('segdataVcut_train.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
del inputs
with h5py.File('segdataVcut_val.hdf5', 'x') as hf:
    hf.create_dataset("inputs",  data=val_inputs,dtype='f')
#%%    
del val_inputs
del aug_inputs
#%% Targets  
mat = spio.loadmat(globfiles[0],squeeze_me=True)
topind = mat['topind']
botind = mat['botind']
numS = topind-botind+1
SegMask = mat['SegMask']
SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
targets = SegMask[botind:topind+1].reshape(numS,256,256,1)

for file in globfiles[1:]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    topind = mat['topind']
    botind = mat['botind']
    numS = topind-botind+1
    SegMask = mat['SegMask']
    SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
    new_targets = SegMask[botind:topind+1].reshape(numS,256,256,1)
    targets = np.concatenate((targets,new_targets),axis=0)
    
    
# synthesize extra data
# LR flips
fl_targets = np.flip(targets,2)

# copy masks for gamma corrections
gm_targets = np.copy(targets)

# combine together
aug_targets = np.concatenate((targets,fl_targets,gm_targets),axis=0)

# pull out validation set
val_targets = np.take(aug_targets,valvec,axis=0)
targets = np.delete(aug_targets,valvec,axis=0)
    
# sort targets same as inputs
print('sorting')
np.take(targets,sort_r,axis=0,out=targets)

print('saving to hdf5')
with h5py.File('segdataVcut_train.hdf5', 'a') as hf:
    hf.create_dataset("targets",  data=targets,dtype='i')
with h5py.File('segdataVcut_val.hdf5', 'a') as hf:
    hf.create_dataset("targets", data=val_targets,dtype='i')
    
print('done')

#%%
del targets
del val_targets
del aug_targets