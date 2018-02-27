# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
import numpy as np
import ants
from keras.models import load_model

datapath = 'RegNIFTIs/subj{:03d}_{}.nii'
outputpath = 'OutputNIFTIs/subj{:03d}_{}.nii'
subj_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
eps = 1e-9
MS = 3 # multislice number

#%% Load model
RegModel = load_model("MuMapModel_best.hdf5",None,False)

#%% Multislice maker for inputs
def ConvertToMultiSlice(array,MS=3):
    tshp = array.shape
    MSos = np.int((MS-1)/2) #MultiSlice Offset
    MSarray = np.zeros((tshp[0],MS,tshp[1],tshp[2],tshp[3]))
    for ss in range(MSos,tshp[0]-MSos):
        MSarray[ss,0,...] = array[ss-1]
        MSarray[ss,1,...] = array[ss]
        MSarray[ss,2,...] = array[ss+1]
    MSarray[0,0,...] = array[0]
    MSarray[0,1,...] = array[0]
    MSarray[0,2,...] = array[1]
    MSarray[-1,0,...] = array[-2]
    MSarray[-1,1,...] = array[-1]
    MSarray[-1,2,...] = array[-1]
    return MSarray
    
#%% Loop over all subjects
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
    inputs = ConvertToMultiSlice(np.stack((wims,fims,inims,outims),axis=3),MS)
    
    #%% Make predictions on MR data
    output = RegModel.predict(inputs,batch_size=16)
    # split outputs into regression and classification
    reg_output = output[0]
    class_output = output[1]
    # get regression output
    # and convert to units of mm
    reg_output[reg_output<0] = 0
    muMaps = reg_output[...,0]/50
    # proces classification output
    class_inds = np.argmax(class_output,axis=3)
    class_muMap = np.zeros(class_inds.shape)
    class_muMap[class_inds==1] = 0.001795 # lungs
    class_muMap[class_inds==2] = 0.01 # soft tissue
    class_muMap[class_inds==3] = 0.0112 # bone
    
    #%% Write output
    # convert to ants images
    nac_img = ants.image_read(datapath.format(subj,'NAC'))
    muMaps = np.rollaxis(muMaps,0,3)
    class_muMap = np.rollaxis(class_muMap,0,3)
    muMap_img = nac_img.new_image_like(muMaps)
    class_img = nac_img.new_image_like(class_muMap)
    # write to nifti file
    ants.image_write(muMap_img,outputpath.format(subj,'regMap'))
    ants.image_write(class_img,outputpath.format(subj,'classMap'))
    print('Completed subject',subj)
    
print('Done!')
