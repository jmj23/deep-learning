#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:04:09 2017

@author: jmj136
"""
import sys
sys.path.insert(0,'/home/jmj136/KerasFiles')
import ants
import numpy as np
from VisTools import multi_slice_viewer0, registration_viewer
from scipy.ndimage import morphology as scimorph
from scipy.ndimage import label, generate_binary_structure

datapath = 'NIFTIs/subj{:03d}_{}.nii'
savepath = 'RegNIFTIs/subj{:03d}_{}.nii'

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
    multi_slice_viewer0(stacked_array,[])
    
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
    nac_img = ants.image_read(datapath.format(subj,'NAC'))
    CT_img = ants.image_read(datapath.format(subj,'CTAC'))
    return water_img,fat_img,inphase_img,nac_img,CT_img
#%% Data processing
def ProcessImgs(nac_img,CT_img):
    # clip NAC image at fixed value
    nac_array = nac_img.numpy()
    nac_array[nac_array>1500] = 1500
    nac_imgC = nac_img.new_image_like(nac_array)
    
    # shift CT image to be all positive
    CT_array = CT_img.numpy()
    CT_array += 1024
    CT_array[CT_array<0]= 0
    CT_imgS = CT_img.new_image_like(CT_array)
    return nac_imgC,CT_imgS

#%% MR-> NAC  Registration
def RegMRNAC(nac_imgC,water_img,fat_img,inphase_img):
    MR_tx = ants.registration(fixed=nac_imgC,moving=water_img,type_of_transform='Affine')
    reg_water = MR_tx['warpedmovout']
    reg_fat = ants.apply_transforms(fixed=nac_imgC, moving=fat_img,
                                    transformlist=MR_tx['fwdtransforms'] )
    reg_inphase = ants.apply_transforms(fixed=nac_imgC, moving=inphase_img,
                                        transformlist=MR_tx['fwdtransforms'] )
    return reg_water,reg_fat,reg_inphase

#%% Mask CT image
def RemoveCTcoil(CT_imgS):
    
    CT_array = CT_imgS.numpy()
    CT_mask = CT_array>500
    CT_mask = np.rollaxis(CT_mask,2,0)
    
    y,x = np.ogrid[-3: 3+1, -3: 3+1]
    strel4 = x**2+y**2 <= 4**2
    strel3 = x**2+y**2 <= 3**2
    s = generate_binary_structure(2,2)
    
    for ii in range(CT_mask.shape[0]):
        CT_mask[ii,...] = scimorph.binary_opening(CT_mask[ii,...],strel4)
        CT_mask[ii,...] = scimorph.binary_closing(CT_mask[ii,...],strel3)
        CT_mask[ii,...] = scimorph.binary_fill_holes(CT_mask[ii,...])
        
        labeled_array, numpatches = label(CT_mask[ii,...],s)
        if numpatches>1:
            sizes = [np.sum(labeled_array==label) for label in range(1,numpatches+1)]
            maxlabel = np.argmax(sizes)+1
            CT_mask[ii,...] = labeled_array==maxlabel
            
    CT_mask = np.rollaxis(CT_mask,0,3)
    CT_array[~CT_mask]= 0
    CT_imgM = CT_imgS.new_image_like(CT_array)
    
    return CT_imgM

#%% CT-> NAC registration
def RegCTNAC(nac_imgC,CT_imgM,reg_water,reg_fat):
    # rigid CT
    CT_rig_reg = ants.registration(fixed=nac_imgC,moving=CT_imgM,type_of_transform='Rigid')
    rig_CT = CT_rig_reg['warpedmovout']
    
    # Create mask for MR registration
    CT_array = rig_CT.numpy()
    maxes = np.max(CT_array,axis=(0,1))
    bad_inds= np.where(maxes==0)
    good_inds= np.where(maxes!=0)[0][1:-1]

    water_array = reg_water.numpy()
    water_array[...,bad_inds] = 0
    fat_array = reg_fat.numpy()
    fat_array[...,bad_inds] = 0
    com_array = water_array+fat_array
    com_imgZ = reg_water.new_image_like(com_array)
    
    # Affine CT
    CT_aff_reg = ants.registration(fixed=com_imgZ,moving=rig_CT,type_of_transform='Affine')
    aff_CT = CT_aff_reg['warpedmovout']
    
    # Affine CT 2
    CT_aff_reg2 = ants.registration(fixed=com_imgZ,moving=CT_imgM,type_of_transform='Affine',
                                    aff_sampling=16)
    aff_CT2 = CT_aff_reg2['warpedmovout']
    
    # nonrigid CT
    CT_syn_reg = ants.registration(fixed=com_imgZ,moving=aff_CT2,type_of_transform='SyN',
                                   syn_sampling=16,reg_iterations=(20,10,0))
    syn_CT = CT_syn_reg['warpedmovout']
    
    # shift CT image back to initial scaling
    CT_array = syn_CT.numpy()
    CT_array -= 1024
    CT_imgRS = syn_CT.new_image_like(CT_array)
    
    return rig_CT,aff_CT,aff_CT2,syn_CT,good_inds,CT_imgRS
#%% Export data
def SaveData(savepath,subj,reg_water,reg_fat,reg_inphase,reg_CT,nac_img,good_inds):
    ants.image_write(reg_water,savepath.format(subj,'WATER'))
    ants.image_write(reg_fat,savepath.format(subj,'FAT'))
    ants.image_write(reg_inphase,savepath.format(subj,'InPhase'))
    ants.image_write(reg_CT,savepath.format(subj,'CTAC'))
    ants.image_write(nac_img,savepath.format(subj,'NAC'))
    np.savetxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj),good_inds,fmt='%u')
#%% Main Script

subjectlist = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#subjectlist = [10]

for subj in subjectlist:
    print('Loading data...')
    water_img,fat_img,inphase_img,nac_img,CT_img = LoadData(datapath,subj)
    print('Processing images...')
    nac_imgC,CT_imgS = ProcessImgs(nac_img,CT_img)
    print('Registering MR images...')
    reg_water,reg_fat,reg_inphase = RegMRNAC(nac_imgC,water_img,fat_img,inphase_img)
    print('Removing coil from CT image...')
    CT_imgM = RemoveCTcoil(CT_imgS)
    print('Registering CT images...')
    rig_CT,aff_CT,aff_CT2,reg_CT,good_inds,CT_imgRS = RegCTNAC(nac_imgC,CT_imgM,reg_water,reg_fat)
#    print('Displaying results...')
#    display_ants([nac_imgC,reg_water,reg_CT])
#    display_ants_reg(reg_water,reg_CT)
#    display_ants([reg_water,rig_CT,aff_CT,aff_CT2,reg_CT])
    print('Saving data for subject',subj,'...')
    SaveData(savepath,subj,reg_water,reg_fat,reg_inphase,CT_imgRS,nac_img,good_inds)
    
