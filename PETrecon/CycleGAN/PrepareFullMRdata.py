#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:59:26 2018

@author: jmj136
"""
import h5py
import numpy as np
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
import ants

datapath = '~/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_{}.nii'
savepath = 'CycleGAN_FullMRdata.hdf5'

subj_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

eps = 1e-12
#%% Inputs
print('Loading inputs')

subj = subj_vec[0]
print('Loading subject',subj)
wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
for im in wims:
    im[im<0]=0
    im /= np.max(im)
for im in fims:
    im[im<0]=0
    im /= np.max(im)
for im in inims:
    im[im<0]=0
    im /= np.max(im)
for im in outims:
    im[im<0]=0
    im /= (np.max(im)+eps)

inputs = np.stack((wims,fims,inims,outims),axis=3)

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
    good_inds = np.loadtxt('/home/jmj136/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    for im in wims:
        im[im<0]=0
        im /= np.max(im)
    for im in fims:
        im[im<0]=0
        im /= np.max(im)
    for im in inims:
        im[im<0]=0
        im /= np.max(im)
    for im in outims:
        im[im<0]=0
        im /= (np.max(im)+eps)
        
    new_inputs = np.stack((wims,fims,inims,outims),axis=3)
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    
print('Storing data as HDF5...')
inputs = np.rot90(inputs,k=1,axes=(1,2))
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset("MR_full",  data=inputs,dtype='f')
    
print('Done!')