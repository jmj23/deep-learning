#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:48:25 2018

@author: jmj136
"""

from glob import glob
import pydicom as dcm
import os
import nibabel as nib
import numpy as np
from operator import itemgetter
#%%
# subjects parent directory
datadir = '/home/jmj136/deep-learning/ItATMIS2/data/RestScans'
# extension of DICOM files
ext = 'dcm'

subj_dirs = glob(os.path.join(datadir, "*", ""))

# current directory
cur_dir = subj_dirs[0]
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
loctimes.sort(key=itemgetter(0))
loctimes.sort(key=itemgetter(1))
# extract sorting indices
sort_inds = [ele[2] for ele in loctimes]
# combine into array with sorting order
image_volume = np.stack([dcms[i].pixel_array for i in sort_inds])

#~#~#~# For 4D case #~#~#~#
# find splitting indices
diffs = np.diff([l[0] for l in loctimes])
spl_inds = np.where(diffs)[0]
img_vols = np.split(image_volume,spl_inds+1)
#~#~#~#~#~#~#~#~#~#~#~#~#~#