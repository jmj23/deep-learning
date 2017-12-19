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
import os

DataDir = 'MatData'
numsubj = 120
numTest = 20
np.random.seed(seed=1)
#%% Inputs
print('Loading inputs')
globfiles = sorted(glob.glob(os.path.join(DataDir,"*.mat")))
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
numSlice = topind-botind
sliceVec = [numSlice]
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
    numSlice = topind-botind
    sliceVec.append(numSlice)
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
    
#eliminate zero masks in targets
targ_sums = np.sum(targets[...,0],axis=(1,2))
zeroinds = np.argwhere(targ_sums==0)
inputs = np.delete(inputs,zeroinds,axis=0)
targets = np.delete(targets,zeroinds,axis=0)
# update slice counts
sliceVec = np.array(sliceVec)
sliceCumArray = np.concatenate((np.array([0]),np.cumsum(sliceVec)),axis=0)
for zz in range(zeroinds.shape[0]):
    zind = zeroinds[zz]
    subj = np.argwhere(sliceCumArray>zind)[0]-1
    sliceVec[subj] = sliceVec[subj]-1
    

#%% Split training, testing data
# get slice ranges corresponding to subjects
sliceCumArray = np.concatenate((np.array([0]),np.cumsum(sliceVec)),axis=0)
# pick out subjects
numSubjs = sliceVec.shape[0]
test_subjs = np.random.choice(np.arange(numSubjs),numTest,replace=False)
# Get testing data
test_inds = np.concatenate(([np.arange(sliceCumArray[ind],sliceCumArray[ind+1]) for ind in test_subjs]))
test_inputs = np.take(inputs,test_inds,axis=0)
test_targets = np.take(targets,test_inds,axis=0)
# Remove from training arrays
inputs = np.delete(inputs,test_inds,axis=0)
targets = np.delete(targets,test_inds,axis=0)

# update slice ranges
sliceVec = np.delete(sliceVec,test_subjs)
sliceCumArray = np.concatenate((np.array([0]),np.cumsum(sliceVec)),axis=0)

# store training and testing data to HDF5 file
print('storing validation data as HDF5')
with h5py.File('DataSelectionData.hdf5', 'x') as hf:
    hf.create_dataset("train_inputs",  data=inputs,dtype='f')
    hf.create_dataset("test_inputs", data=test_inputs,dtype='f')
    hf.create_dataset("train_targets",  data=targets,dtype='i')
    hf.create_dataset("test_targets",  data=test_targets,dtype='i')
    hf.create_dataset("CumSliceArray",data=sliceCumArray,dtype='i')
del test_inputs
del test_targets
del inputs
del targets
