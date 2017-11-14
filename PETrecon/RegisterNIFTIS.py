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

datapath = 'NIFTIs/subj{:03d}_{}.nii'

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
    registration_viewer(nparray1,nparray2,.5)

#%% Data loading
subj = 2
water_img = ants.image_read(datapath.format(subj,'WATER'))
fat_img = ants.image_read(datapath.format(subj,'FAT'))
inphase_img = ants.image_read(datapath.format(subj,'InPhase'))
nac_img = ants.image_read(datapath.format(subj,'NAC'))
CT_img = ants.image_read(datapath.format(subj,'CT'))

#%% Data processing
# clip NAC image at fixed value
nac_array = nac_img.numpy()
nac_array[nac_array>2500] = 2500
nac_imgC = nac_img.new_image_like(nac_array)

# shift CT image to be all positive
CT_array = CT_img.numpy()
CT_array += 1024
CT_array[CT_array<0]= 0
CT_imgS = CT_img.new_image_like(CT_array)

# make combined MR image
fat_array = fat_img.numpy()
water_array = water_img.numpy()
com_array = fat_array+water_array
com_img = water_img.new_image_like(com_array)


#%% MR-> NAC  Registration
MR_tx = ants.registration(fixed=nac_imgC,moving=water_img,type_of_transform='Affine')
reg_water = MR_tx['warpedmovout']
reg_fat = ants.apply_transforms(fixed=nac_imgC, moving=fat_img,
                                transformlist=MR_tx['fwdtransforms'] )
reg_inphase = ants.apply_transforms(fixed=nac_imgC, moving=inphase_img,
                                    transformlist=MR_tx['fwdtransforms'] )

#%% create mask for CT registration
NAC_tx = ants.registration(fixed=water_img,moving=nac_imgC,type_of_transform='Affine')
nac_mr_array = NAC_tx['warpedmovout'].numpy()
nac_mr_array -= np.min(nac_mr_array)
nac_mr_array /= np.max(nac_mr_array)
nac_mask = nac_mr_array>.06
nac_mask_img = water_img.new_image_like(nac_mask)

#%% CT-> MR registration
CT_aff = ants.registration(fixed=com_img,moving=CT_imgS,type_of_transform='Affine')
CT_syn = ants.registration(fixed=com_img,moving=CT_imgS,type_of_transform='SyN',
                           initial_transform=CT_aff['fwdtransforms'])
# affine CT
aff_CT = ants.apply_transforms(fixed=com_img,moving=CT_imgS,
                               transformlist=CT_aff['fwdtransforms'])
# non-Rigid CT
syn_CT = CT_syn['warpedmovout']


# CT-> MR-> NAC
reg_CT = ants.apply_transforms(fixed=nac_imgC, moving=aff_CT,
                                transformlist=MR_tx['fwdtransforms'])
reg_CT2 = ants.apply_transforms(fixed=nac_imgC, moving=syn_CT,
                                transformlist=MR_tx['fwdtransforms'])

display_ants([nac_imgC,reg_water,reg_CT,reg_CT2])
display_ants([water_img,com_img,aff_CT,syn_CT])
display_ants_reg(nac_imgC,reg_CT)