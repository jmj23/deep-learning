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

MRpath = 'NIFTIs/subj{:03d}_{}.nii'
ACpath = 'BREAST_RECON/fulldose/volunteer{:03d}_fulldose.nii.gz'
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
def LoadData(MRpath,ACpath,subj):
    water_img = ants.image_read(MRpath.format(subj,'WATER'))
    fat_img = ants.image_read(MRpath.format(subj,'FAT'))
    AC_img = ants.image_read(ACpath.format(subj))
    AC_array = AC_img.numpy()
    AC_array[AC_array>10000] = 10000
    AC_array /= 10000
    AC_imgP = AC_img.new_image_like(AC_array)
    
    return water_img,fat_img,AC_imgP
#%% MR-> NAC  Registration
def RegMRNAC(AC_img,water_img,fat_img):
    MR_tx = ants.registration(fixed=AC_img,moving=water_img,type_of_transform='Similarity')
    reg_water = MR_tx['warpedmovout']
    reg_fat = ants.apply_transforms(fixed=AC_img, moving=fat_img,
                                    transformlist=MR_tx['fwdtransforms'])
    return reg_water,reg_fat


#%% Export data
def SaveData(savepath,subj,reg_water,reg_fat):
    ants.image_write(reg_water,savepath.format(subj,'WATER'))
    ants.image_write(reg_fat,savepath.format(subj,'FAT'))
    
#%% Main Script

subjectlist = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#subjectlist = [5]

for subj in subjectlist:
    print('Loading data...')
    water_img,fat_img,AC_img= LoadData(MRpath,ACpath,subj)
    
    print('Registering MR images...')
    reg_water,reg_fat = RegMRNAC(AC_img,water_img,fat_img)
    
#    print('Displaying results...')
#    display_ants([reg_water,AC_img,reg_fat])
#    display_ants_reg(AC_img,reg_water)
    print('Saving data for subject',subj,'...')
    SaveData(savepath,subj,reg_water,reg_fat)
    