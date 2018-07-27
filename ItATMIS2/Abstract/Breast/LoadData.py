#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:57:31 2018

@author: jmj136
"""
import os
import numpy as np
import h5py
import pydicom as dcm
from glob import glob
from skimage.draw import polygon
from operator import itemgetter
#%% directories
data_dir = '/data/jmj136/ItATMIS/CAP/SCD_DeidentifiedImages/'
annote_dir = '/data/jmj136/ItATMIS/CAP/SCD_ManualContours/'
# SC-HF-I-01/contours-manual/IRCCI-expert
savepath = '/data/jmj136/ItATMIS/CAP/data.hdf5'

# find subject directories
subj_dirs = glob(os.path.join(data_dir, "*", ""))

# loop over subjects
subj = 0
# current subject directory
cur_dir = subj_dirs[0]
# get list of dicom files
dcm_files = []
for root, dirs, files in os.walk(cur_dir):
    dcm_files = dcm_files + [(os.path.join(root,f)) for f in files if f.endswith('.dcm')]

dcm_dirs = glob(os.path.join(cur_dir,'*'))
dcm_files = glob(os.path.join(dcm_dirs[0],'*.dcm'))
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



 # Z positions
z = [d.ImagePositionPatient[2] for d in dcms]
# Rows and columns
pos_r = dcms[0].ImagePositionPatient[1]
spacing_r = dcms[0].PixelSpacing[1]
pos_c = dcms[0].ImagePositionPatient[0]
spacing_c = dcms[0].PixelSpacing[0]
# Preallocate
mask = np.zeros_like(ims)
# loop over the different slices that each contour is on
nodes = np.array(c).reshape((-1, 3))
assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
zNew = [round(elem,1) for elem in z]
try:
    z_index = z.index(nodes[0,2])
except ValueError:
    z_index = zNew.index(nodes[0,2])
r = (nodes[:, 1] - pos_r) / spacing_r
c = (nodes[:, 0] - pos_c) / spacing_c
rr, cc = polygon(r, c)
mask[z_index,rr, cc] = 1
    
with h5py.File(savepath, 'w') as hf:
    dataname = 'subj_{:03d}_mask'.format(subj)
    hf.create_dataset(dataname,  data=masks, dtype='f')
    dataname = 'subj_{:03d}_image'.format(subj)
    hf.create_dataset(dataname,  data=images, dtype='f')