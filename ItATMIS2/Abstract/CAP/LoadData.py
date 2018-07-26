#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:57:31 2018

@author: jmj136
"""
import numpy as np
import h5py
import pydicom as dcm
from glob import glob
from skimage.draw import polygon
#%%
data_dir = '/data/jmj136/ItATMIS/CAP/SCD_deidentifiedimages/'
annote_dir = '/data/jmj136/ItATMIS/CAP/SCD_ManualContours/'
# SC-HF-I-01/contours-manual/IRCCI-expert
savepath = '/data/jmj136/ItATMIS/CAP/data.hdf5'

subj = 0




 # Z positions
z = [d.ImagePositionPatient[2] for d in dicms]
# Rows and columns
pos_r = dicms[0].ImagePositionPatient[1]
spacing_r = dicms[0].PixelSpacing[1]
pos_c = dicms[0].ImagePositionPatient[0]
spacing_c = dicms[0].PixelSpacing[0]
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