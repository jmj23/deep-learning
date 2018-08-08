#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:26:18 2018

@author: jmj136
"""
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
sys.path.insert(1,'/home/jmj136/deep-learning/ItATMIS2')
import numpy as np
from scipy import io
from glob import glob
from natsort import natsorted
from ItATMISfunctions import BlockModel, GetLCTSCdata, GetCardiacData
from VisTools import DisplayDifferenceMask
import os
import GPUtil
try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except RuntimeError as e:
    print('No GPU available')
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
#%% Setup
# Data directories
breast_data_dir = '/home/jmj136/Data/ItATMIS/Breast'
breast_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Breast/best_model_weights.h5'
lung_data_dir = '/home/jmj136/Data/ItATMIS/LCTSC'
lung_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/LCTSC/best_model_weights.h5'
cardiac_data_dir = '/home/jmj136/Data/ItATMIS/Cardiac'
cardiac_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Cardiac/best_model_weights.h5'
image_save_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/{}Sample.png'

# subjects to use (breast,lung,cardiac)
subjs = [6,5,5]

#%% Breast data
breast_files = natsorted(glob(os.path.join(breast_data_dir,'*.mat')))
mat = io.loadmat(breast_files[subjs[0]])    
breast_x = mat['InputArray']
breast_y = mat['TargetArray'][...,0]
breast_slice = 13
# create model and load trained weights
breast_model = BlockModel(breast_x.shape,filt_num=16,numBlocks=4,num_out_channels=1)
breast_model.load_weights(breast_weights_path)
# predict on inputs
breast_pred = breast_model.predict(breast_x)
# get slices of interest
breast_im = breast_x[breast_slice,...,0]
breast_mask_truth = breast_y[breast_slice,...]
breast_mask_pred = breast_pred[breast_slice,...,0]
# make side-by-side image
breast_joint_im = np.c_[breast_im,breast_im]
breast_joint_mask1 = np.c_[0*breast_mask_truth,breast_mask_truth]
breast_joint_mask2 = np.c_[0*breast_mask_pred,breast_mask_pred]
# save path
lung_save_path = image_save_path.format('Breast')
# display and save image
DisplayDifferenceMask(breast_joint_im,
                      breast_joint_mask1,
                      breast_joint_mask2,
                      name='Breast Difference Mask',
                      savepath=lung_save_path)


#%% Lung data
subj_dirs = glob(os.path.join(lung_data_dir,'LCTSC-Train*'))
# slice to display
lung_slice = 74
data = GetLCTSCdata(subj_dirs[subjs[1]])
# separate into inputs and targets
lung_x = data[0]
lung_y = data[1]
# create model and load trained weights
lung_model = BlockModel(lung_x.shape,filt_num=16,numBlocks=4,num_out_channels=1)
lung_model.load_weights(lung_weights_path)
# predict on inputs
lung_pred = lung_model.predict(lung_x)
# get slices of interest
lung_im = lung_x[lung_slice,...,0]
lung_mask_truth = lung_y[lung_slice,...,0]
lung_mask_pred = lung_pred[lung_slice,...,0]
# make side-by-side image
lung_joint_im = np.c_[lung_im,lung_im]
lung_joint_mask1 = np.c_[0*lung_mask_truth,lung_mask_truth]
lung_joint_mask2 = np.c_[0*lung_mask_pred,lung_mask_pred]
# save path
lung_save_path = image_save_path.format('Lung')
# display and save image
DisplayDifferenceMask(lung_joint_im,
                      lung_joint_mask1,
                      lung_joint_mask2,
                      name='Lung Difference Mask',
                      savepath=lung_save_path)

#%% Cardiac Data
image_files = natsorted(glob(os.path.join(cardiac_data_dir,'sol*.mat')))
contour_files = natsorted(glob(os.path.join(cardiac_data_dir,'man*.mat')))
subjs = range(len(image_files))
cardiac_slice = 6
data = GetCardiacData(image_files[subjs[2]],contour_files[subjs[2]])
# separate into inputs and targets
cardiac_x = data[0]
cardiac_y = data[1]
# create model and load trained weights
cardiac_model = BlockModel(cardiac_x.shape,filt_num=16,numBlocks=4,num_out_channels=1)
cardiac_model.load_weights(cardiac_weights_path)
# predict on inputs
cardiac_pred = cardiac_model.predict(cardiac_x)
# get slices of interest
cardiac_im = cardiac_x[cardiac_slice,...,0]
cardiac_mask_truth = cardiac_y[cardiac_slice,...,0]
cardiac_mask_pred = cardiac_pred[cardiac_slice,...,0]
# make side-by-side image
cardiac_joint_im = np.c_[cardiac_im,cardiac_im]
cardiac_joint_mask1 = np.c_[0*cardiac_mask_truth,cardiac_mask_truth]
cardiac_joint_mask2 = np.c_[0*cardiac_mask_pred,cardiac_mask_pred]
# save path
cardiac_save_path = image_save_path.format('Cardiac')
# display and save image
DisplayDifferenceMask(cardiac_joint_im,
                      cardiac_joint_mask1,
                      cardiac_joint_mask2,
                      name='Cardiac Difference Mask',
                      savepath=cardiac_save_path)
