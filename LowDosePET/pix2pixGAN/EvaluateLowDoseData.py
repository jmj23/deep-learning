#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 09:25:28 2018

@author: jmj136
"""
import os
# Use first available GPU
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import ants
import nibabel as nib
from keras.models import load_model

LDtype = '30s'
MS = 5
MSos = 2

LowDosePath = '../lowdose_{:s}/volunteer{{:03d}}_lowdose.nii.gz'.format(LDtype)
waterpath = '../RegNIFTIs/subj{:03d}_WATER.nii'
fatpath = '../RegNIFTIs/subj{:03d}_FAT.nii'
outputpath = 'OutputNIFTIs/subj{{:03d}}_{}_simFD.nii'.format(LDtype)
model_filepath = 'LowDosePET_pix2pixModel_{:s}.h5'.format(LDtype)


subj_vec = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
eps = 1e-9
normfac = 20000

#%% Load model
LowDoseModel = load_model(model_filepath,None,False)

#%% Multislice converter
def ConvertToMultiSlice(array,MS=3):
    # input shape
    tshp = array.shape
    MSos = np.int((MS-1)/2) #MultiSlice Offset
    # pre allocate
    MSarray = np.zeros((tshp[0],MS,tshp[1],tshp[2],tshp[3]))
    # loop over main body of array
    for ss in range(tshp[0]):
        for mm in range(MS):
            ind = np.clip(ss-MSos+mm,0,tshp[0]-1)
            MSarray[ss,mm,...] = array[ind]        
    return MSarray
    
#%% Loop over all subjects
for ii in range(len(subj_vec)):
    # Inputs
    subj = subj_vec[ii]
    print('Processing subject',subj)
    # load water nifti and normalize
    wnft = nib.as_closest_canonical(nib.load(waterpath.format(subj)))
    wims = np.flip(np.rot90(np.rollaxis(wnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    fnft = nib.as_closest_canonical(nib.load(fatpath.format(subj)))
    fims= np.flip(np.rot90(np.rollaxis(fnft.get_data(),2,0),k=-1,axes=(1,2)),2)
    for im in wims:
        im[im<0]=0
        im /= np.max(im)+eps
    for im in fims:
        im[im<0]=0
        im /= np.max(im)+eps
    LDims = nib.as_closest_canonical(nib.load(LowDosePath.format(subj))).get_data()[...,0]
    LDims = np.rollaxis(np.rollaxis(LDims,2,0),2,1)
    for im in LDims:
        im[im<0]=0
        im /= normfac
    
    inputs = np.stack((LDims,wims,fims),axis=3)
    inputs = ConvertToMultiSlice(inputs,MS)
    
    # Make predictions on Low Dose data
    output = LowDoseModel.predict(inputs,batch_size=16)
    
    # Write output
    PETims = np.rot90(np.rollaxis(output,0,3),-1)
    # write to nifti file
    PETims = normfac*PETims[...,0]
    output_img = nib.Nifti1Image(PETims, np.eye(4))
    output_img.to_filename(outputpath.format(subj))
    print('Completed subject',subj)
    
print('Done!')