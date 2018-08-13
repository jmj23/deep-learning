#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:48:25 2018

@author: jmj136
"""

import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from glob import glob
import pydicom as dcm
import os
import nibabel as nib
import numpy as np
from operator import itemgetter
#%%
# subjects parent directory
datadir = '/home/jmj136/Data/AlanCardiacData/RestScans'
# output directory
outputpath = '/home/jmj136/Data/AlanCardiacData/RestNiftis/subj{}_time{}'
# extension of DICOM files
ext = 'dcm'

subj_dirs = glob(os.path.join(datadir, "*", ""))
for subj,cur_dir in enumerate(subj_dirs):
    print('Processing subject {}...'.format(subj))
    # get list of files
    dcm_files = glob(os.path.join(cur_dir,"*."+ext))
    # load dicom files
    dcms = [dcm.read_file(f) for f in dcm_files]
    # prepare sorting list of tuples
    dcm_inds = list(range(len(dcms)))
    locs = [(float(d.SliceLocation)) for d in dcms]
    times = [(int(d.InstanceNumber)) for d in dcms]
    loctimes = list(zip(locs,times,dcm_inds))
    # sort by time then slice
    loctimes.sort(key=itemgetter(1))
    # extract sorting indices
    sort_inds = [ele[2] for ele in loctimes]
    # combine into array with sorting order
    image_volume = np.stack([dcms[i].pixel_array for i in sort_inds])
    # find number of time points per slice
    sort_locs = [ele[0] for ele in loctimes]
    slice_changes = np.where(np.diff(sort_locs))[0]
    # split into 4D by time points
    image_4D_volume = np.stack(np.split(image_volume,slice_changes+1))
    # put time points 1st and slices 2nd axes
    image_4D_volume = np.rollaxis(image_4D_volume,1,0)
    # create nifti data
    for t,im_vol in enumerate(image_4D_volume):
        if t <= 10:
            vol = np.moveaxis(im_vol,0,-1)
            img = nib.Nifti1Image(vol,np.eye(4))
            img.to_filename(outputpath.format(subj,t))