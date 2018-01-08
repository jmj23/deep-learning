# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import ants
from keras.models import load_model
from CustomMetrics import weighted_mae

datapath = 'NIFTIs/subj{:03d}_{}.nii'
outputpath = 'OutputNIFTIs/subj{:03d}_{}.nii'
subj_vec = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
eps = 1e-9

#%% Load model
RegModel = load_model("LowDosePETModel_best.hdf5",
                      custom_objects={'weighted_mae':weighted_mae})
    
# Loop over all subjects
for ii in range(len(subj_vec)):
    #%% Inputs
    subj = subj_vec[ii]
    print('Processing subject',subj)
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
    good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
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
    
    #%% Make predictions on Low Dose data
    output = RegModel.predict(inputs,batch_size=16)
    PETims = output
    
    #%% Write output
    # convert to ants images
    nac_img = ants.image_read(datapath.format(subj,'NAC'))
    PETims = np.rollaxis(PETims,0,3)
    PET_img = nac_img.new_image_like(PETims)
    # write to nifti file
    ants.image_write(PET_img,outputpath.format(subj,'DL_PETims'))
    print('Completed subject',subj)
    
print('Done!')
