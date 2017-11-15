# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
# This function gives an example of how to load
# segmentation training data from .mat files
# into python and save it to a python friendly
# format: .hdf5
# The data can then be used with
# SegmentationTraining.py
# to create a segmentation deep learning model

import h5py
import numpy as np
import glob
import scipy.io as spio

# Directory that the .mat files are in
# adjust as needed
DataDir = '/home/jmj136/KerasFiles/MatData'

# file name for output
savename = 'SegData.hdf5'

#%% Inputs
# Get the list of all .mat files in that directory
globfiles = sorted(glob.glob(DataDir+"/*.mat"))
# Load first .mat file as a starting point
mat = spio.loadmat(globfiles[0],squeeze_me=True)
# Load water images- this is an example of how to reference
# variables in .mat files
waterims = mat['imagesW']
# MATLAB 3D arrays are generally written as (row,col,slice)
# but python works better with (slice,row,col) so we will
# reformat here
waterims = np.rollaxis(waterims.astype(np.float32, copy=False),2,0)
# repeat for fat images, in this case
fatims = mat['imagesF']
fatims = np.rollaxis(fatims.astype(np.float32, copy=False),2,0)
# This example application uses a 3rd channel input, Vmask.
# It acts the same as any other input channel
Vcut = mat['Vmask']
Vcut = np.rollaxis(Vcut.astype(np.float32, copy=False),2,0)
# This example has bottom and top indices stored to reference
# which slices to use
topind = mat['topind']
botind = mat['botind']
# Normalize each slice
for im in waterims:
    im /= np.max(im)
for im in fatims:
    im /= np.max(im)
# grab the correct slices from each image volume
numS = topind-botind+1
waterims = waterims[botind:topind+1]
fatims = fatims[botind:topind+1]
Vcut = Vcut[botind:topind+1]
# stack all the channels together in a new axis,
# forming an array that is (slice,row,col,channel)
inputs = np.stack((waterims,fatims,Vcut),axis=-1)

# now repeat these steps for all of the other files
# and concatenate the inputs together along the slice axis
# (axis 0)
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
    
#%% Targets  
# Now we will load the corresponding targets (segmentation volumes)
# These are stored in the same files as separate variables
mat = spio.loadmat(globfiles[0],squeeze_me=True)
topind = mat['topind']
botind = mat['botind']
SegMask = mat['SegMask']
SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
targets = SegMask[botind:topind+1]
# Targets must have same number of dimensions as inputs,
# i.e. (slice,row,column,channel) even if we just have
# a single channel. Add a singular channel using np.newaxis
targets = targets[...,np.newaxis]

# repeat for all the files and concatenate
for file in globfiles[1:]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    topind = mat['topind']
    botind = mat['botind']
    SegMask = mat['SegMask']
    SegMask = np.rollaxis(SegMask.astype(np.int, copy=False),2,0)
    new_targets = SegMask[botind:topind+1]
    new_targets = new_targets[...,np.newaxis]
    targets = np.concatenate((targets,new_targets),axis=0)

#%% store to HDF5 file
print('storing as HDF5')
with h5py.File(savename, 'x') as hf:
    hf.create_dataset("x",  data=inputs,dtype='f')
    hf.create_dataset("y",  data=targets,dtype='i')
    
print('done')