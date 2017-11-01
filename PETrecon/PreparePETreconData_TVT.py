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

DataDir = 'MatData'
filepath = "{}/segdata{:03d}.mat".format(DataDir, 1)
numsubj = 120
val_split = .2
test_split = .2
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
topind = mat['topind']
botind = mat['botind']
for im in waterims:
    im /= np.max(im)
for im in fatims:
    im /= np.max(im)

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
    
    waterims = waterims[botind:topind]
    fatims = fatims[botind:topind]
    Vcut = Vcut[botind:topind]
    new_inputs = np.stack((waterims,fatims,Vcut),axis=-1)
    inputs = np.concatenate((inputs,new_inputs),axis=0)

#%% Targets
print('Loading targets')
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
    
#%% Calculate ranking and split validation,testing data

#eliminate zero masks in targets
targ_sums = np.sum(targets[...,0],axis=(1,2))
zeroinds = np.argwhere(targ_sums==0)
inputs = np.delete(inputs,zeroinds,axis=0)
targets = np.delete(targets,zeroinds,axis=0)

with np.errstate(divide='ignore', invalid='ignore'):
    PDWF = np.nan_to_num(np.divide(inputs[...,0],(inputs[...,0]+inputs[...,1])))
PDWFvals = np.sum(targets[...,0]*PDWF,axis=(1,2))/np.sum(targets[...,0],axis=(1,2))
sortorder = np.argsort(PDWFvals)

valrange = np.round(np.arange(0,len(sortorder),1/val_split)).astype(np.int)
val_inds = sortorder[valrange]

val_inputs = np.take(inputs,val_inds,axis=0)
val_targets = np.take(targets,val_inds,axis=0)

inputs = np.delete(inputs,val_inds,axis=0)
targets = np.delete(targets,val_inds,axis=0)
PDWFvals = np.delete(PDWFvals,val_inds)
sortorder = np.argsort(PDWFvals)

testrange = np.round(np.arange(0,len(sortorder),1/test_split)).astype(np.int)
test_inds = sortorder[testrange]

test_inputs = np.take(inputs,test_inds,axis=0)
test_targets = np.take(targets,test_inds,axis=0)

inputs = np.delete(inputs,test_inds,axis=0)
targets = np.delete(targets,test_inds,axis=0)

# store validation and testing data to HDF5 file
print('storing validation data as HDF5')
with h5py.File('segdata_tvt.hdf5', 'x') as hf:
    hf.create_dataset("val_inputs",  data=val_inputs,dtype='f')
    hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
with h5py.File('segdata_tvt.hdf5', 'a') as hf:
    hf.create_dataset("val_targets",  data=val_targets,dtype='i')
    hf.create_dataset("test_targets",  data=test_targets,dtype='i')
del val_inputs
del val_targets
del test_inputs
del test_targets

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
del inputs
del fl_inputs
del gm_inputs
del targets
del fl_targets
del gm_targets

#%% finalize training data

# randomize inputs
print('randomizing training inputs')
numS = aug_inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(aug_inputs,sort_r,axis=0,out=aug_inputs)
np.take(aug_targets,sort_r,axis=0,out=aug_targets)


# store training data
print('storing train data as HDF5')
with h5py.File('segdata_tvt.hdf5', 'a') as hf:
    hf.create_dataset("train_inputs",  data=aug_inputs,dtype='f')
with h5py.File('segdata_tvt.hdf5', 'a') as hf:
    hf.create_dataset("train_targets",  data=aug_targets,dtype='i')
    
print('done')

del aug_inputs
del aug_targets
