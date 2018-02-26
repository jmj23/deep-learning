#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:04:09 2017

@author: jmj136
"""
import sys
sys.path.insert(0,'/home/jmj136/deep-learning')
import ants
import numpy as np
from VisTools import multi_slice_viewer0, registration_viewer
from scipy.ndimage import morphology as scimorph
#from skimage.morphology import diamond as diastrel
from scipy.ndimage import label, generate_binary_structure

datapath = 'NIFTIs/subj{:03d}_{}.nii'
savepath = 'CycleRegNIFTIs/subj{:03d}_{}.nii'

#%% Display methods
def display_ants(imglist):
    try:
        numI = len(imglist)
    except Exception as e:
        numI = 1
        imglist = [imglist]
    arraylist = []
    for ii in range(numI):
        nparray = imglist[ii].numpy()
        nparray -= np.min(nparray)
        nparray /= np.max(nparray)
        nparray = np.rollaxis(np.rollaxis(nparray,2,0),2,1)
        arraylist.append(nparray)
    stacked_array = np.dstack(arraylist)
    multi_slice_viewer0(stacked_array)
    
def display_ants_reg(img1,img2):
    nparray1 = img1.numpy()
    nparray1 -= np.min(nparray1)
    nparray1 /= np.max(nparray1)
    nparray1 = np.rollaxis(np.rollaxis(nparray1,2,0),2,1)
    nparray2 = img2.numpy()
    nparray2 -= np.min(nparray2)
    nparray2 /= np.max(nparray2)
    nparray2 = np.rollaxis(np.rollaxis(nparray2,2,0),2,1)    
    registration_viewer(nparray1,nparray2,.6)

#%% Data loading
def LoadData(datapath,subj):
    water_img = ants.image_read(datapath.format(subj,'WATER'))
    fat_img = ants.image_read(datapath.format(subj,'FAT'))
    inphase_img = ants.image_read(datapath.format(subj,'InPhase'))
    outphase_img = ants.image_read(datapath.format(subj,'OutPhase'))
    nac_img = ants.image_read(datapath.format(subj,'NAC'))
    CT_img = ants.image_read(datapath.format(subj,'CTAC'))
    CTmac_img = ants.image_read(datapath.format(subj,'CTMAC'))
    return water_img,fat_img,inphase_img,outphase_img,nac_img,CT_img,CTmac_img
#%% Data processing
def ProcessImgs(nac_img,CT_img,CTmac_img):
    # clip NAC images at fixed value
    nac_array = nac_img.numpy()
    nac_array[nac_array>1500] = 1500
    nac_imgC = nac_img.new_image_like(nac_array)
    
    CTmac_array = CTmac_img.numpy()
    CTmac_array[CTmac_array>1500] = 1500
    CTmac_imgC = CTmac_img.new_image_like(CTmac_array)
    
    # shift CT image to be all positive
    CT_array = CT_img.numpy()
    CT_array += 1024
    CT_array[CT_array<0]= 0
    CT_imgS = CT_img.new_image_like(CT_array)
    return nac_imgC,CT_imgS,CTmac_imgC

#%% MR-> NAC  Registration
def RegMRNAC(nac_imgC,water_img,fat_img,inphase_img,outphase_img):
    MR_tx = ants.registration(fixed=nac_imgC,moving=water_img,type_of_transform='Similarity')
    reg_water = MR_tx['warpedmovout']
    reg_fat = ants.apply_transforms(fixed=nac_imgC, moving=fat_img,
                                    transformlist=MR_tx['fwdtransforms'])
    reg_inphase = ants.apply_transforms(fixed=nac_imgC, moving=inphase_img,
                                        transformlist=MR_tx['fwdtransforms'])
    reg_outphase = ants.apply_transforms(fixed=nac_imgC, moving=outphase_img,
                                        transformlist=MR_tx['fwdtransforms'])
    return reg_water,reg_fat,reg_inphase, reg_outphase

#%% Mask CT image
def RemoveCTcoil(CT_imgS,CT_img,CTmac_imgC):
    
    CT_array = np.rollaxis(CT_imgS.numpy(),2,0)
    CT_mask = CT_array>300
    
    # register CTmac to CT
    CTmac_rig_reg = ants.registration(fixed=CT_imgS,moving=CTmac_imgC,type_of_transform='Rigid')
    rig_mac = CTmac_rig_reg['warpedmovout']
    mac_array = np.rollaxis(rig_mac.numpy(),2,0)
    mac_mask = mac_array>350
    
    y,x = np.ogrid[-5: 5+1, -5: 5+1]
    strel5 = x**2+y**2 <= 5**2
    y,x = np.ogrid[-4: 4+1, -4: 4+1]
    strel4 = x**2+y**2 <= 4**2
    s = generate_binary_structure(2,2)
    
    for ii in range(CT_mask.shape[0]):
        mac_mask[ii,...] = scimorph.binary_opening(mac_mask[ii,...],strel4)
        CT_mask[ii,...] = scimorph.binary_opening(CT_mask[ii,...],strel4)
        mac_mask[ii,...] = scimorph.binary_closing(mac_mask[ii,...],strel5)
        mac_mask[ii,...] = scimorph.binary_fill_holes(mac_mask[ii,...])
        CT_mask[ii,...] = scimorph.binary_fill_holes(CT_mask[ii,...])
        
        labeled_array, numpatches = label(mac_mask[ii,...],s)
        if numpatches>1:
            sizes = [np.sum(labeled_array==label) for label in range(1,numpatches+1)]
            maxlabel = np.argmax(sizes)+1
            mac_mask[ii,...] = labeled_array==maxlabel
        labeled_array, numpatches = label(CT_mask[ii,...],s)
        if numpatches>1:
            sizes = [np.sum(labeled_array==label) for label in range(1,numpatches+1)]
            maxlabel = np.argmax(sizes)+1
            CT_mask[ii,...] = labeled_array==maxlabel
    
    comb_mask = CT_mask*mac_mask
    comb_mask = np.rollaxis(comb_mask,0,3)
    CT_array = np.rollaxis(CT_array,0,3)
    CT_array[~comb_mask]= 0
    CT_imgM = CT_imgS.new_image_like(CT_array)
    
    return CT_imgM,rig_mac

#%% CT-> NAC registration
def RegCTNAC(nac_imgC,CT_imgM,reg_water,reg_fat,CTmac_imgC,rig_mac):
    # rigid CT
    MACtx = ants.registration(fixed=nac_imgC,moving=rig_mac,type_of_transform='Similarity')
    rig_CT = ants.apply_transforms(fixed=nac_imgC, moving=CT_imgM,
                                    transformlist=MACtx['fwdtransforms'] )
    
    # shift CT image back to initial scaling
    CT_array = rig_CT.numpy()
    CT_array -= 1024
    CT_imgRS = rig_CT.new_image_like(CT_array)
    
    return rig_CT,CT_imgRS
#%% Export data
def SaveData(savepath,subj,reg_water,reg_fat,reg_inphase,reg_outphase,reg_CT,nac_img):
    ants.image_write(reg_water,savepath.format(subj,'WATER'))
    ants.image_write(reg_fat,savepath.format(subj,'FAT'))
    ants.image_write(reg_inphase,savepath.format(subj,'InPhase'))
    ants.image_write(reg_outphase,savepath.format(subj,'OutPhase'))
    ants.image_write(reg_CT,savepath.format(subj,'CTAC'))
    ants.image_write(nac_img,savepath.format(subj,'NAC'))
    
#%% Main Script

subjectlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
#subjectlist = [1]

for subj in subjectlist:
    print('Loading data...')
    water_img,fat_img,inphase_img,outphase_img,nac_img,CT_img,CTmac_img = LoadData(datapath,subj)
    print('Processing images...')
    nac_imgC,CT_imgS,CTmac_imgC = ProcessImgs(nac_img,CT_img,CTmac_img)
    print('Registering MR images...')
    reg_water,reg_fat,reg_inphase,reg_outphase = RegMRNAC(nac_imgC,water_img,fat_img,inphase_img,outphase_img)
    print('Removing coil from CT image...')
    CT_imgM,rig_mac = RemoveCTcoil(CT_imgS,CT_img,CTmac_imgC)
    print('Registering CT images...')
    reg_CT,CT_imgRS = RegCTNAC(nac_imgC,CT_imgM,reg_water,reg_fat,CTmac_imgC,rig_mac)
#    print('Displaying results...')
#    display_ants([nac_imgC,reg_water,reg_CT])
#    display_ants_reg(reg_water,reg_CT)
    print('Saving data for subject',subj,'...')
    SaveData(savepath,subj,reg_water,reg_fat,reg_inphase,reg_outphase,CT_imgRS,nac_img)
    