#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:15:45 2018

@author: jmj136
"""
#%%
import glob
import pydicom as dcm
import os
import numpy as np
from natsort import natsorted
from matplotlib import pyplot as plt
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from VisTools import slice_viewer4D
data_dir = '/home/jmj136/Data/ItATMIS/CAP'
#%%
def GetDcmVolume(directory):
    files = natsorted(glob.glob(os.path.join(directory,"*.dcm")))
    dicoms = [dcm.dcmread(f) for f in files]
    images = np.stack([d.pixel_array for d in dicoms])
    return images    
#%%
cur_dirs = glob.glob(os.path.join(data_dir, "*", ""))
image_dir = cur_dirs[0]
contour_dir = cur_dirs[1]
subj_dirs = natsorted(glob.glob(os.path.join(image_dir, "*", "")))

cur_subj_dir = subj_dirs[0]

cur_dirs = natsorted(glob.glob(os.path.join(cur_subj_dir, "*-z_+x_256*", "")))

images = np.stack([GetDcmVolume(d) for d in cur_dirs[:7]])
slice_viewer4D(images/1000)

#%%
subj_dirs = natsorted(glob.glob(os.path.join(contour_dir, "*", "")))
cur_subj_dir = subj_dirs[0]

cur_dir = natsorted(glob.glob(os.path.join(cur_subj_dir, "*", "")))[0]

cur_dir = natsorted(glob.glob(os.path.join(cur_dir, "*", "")))[0]

files = natsorted(glob.glob(os.path.join(cur_dir, "*")))

test = np.loadtxt(files[0])
pos_r = 0
pos_c = 0
spacing_r = 1.367188
spacing_c = 1.367188
r = (test[:, 1] - pos_r) / spacing_r
c = (test[:, 0] - pos_c) / spacing_c
plt.plot(c,r,'-')

for f in files[1:3]:
    test = np.loadtxt(f)
    plt.plot(test[:,0],test[:,1],'-')
    
#%%
dcm_files = []
for root, dirs, files in os.walk(cur_subj_dir):
    dcm_files = dcm_files + [(os.path.join(root,f)) for f in files if f.endswith('.dcm')]
    
dcms = [dcm.dcmread(f) for f in dcm_files]

dcm_inds = list(range(len(dcms)))
locs = [(float(d.SliceLocation)) for d in dcms]
times = [(int(d.InstanceNumber)) for d in dcms]
rows = [(int(d.Rows)) for d in dcms]
spacing = [d.PixelSpacing for d in dcms]
position = [d.ImagePositionPatient for d in dcms]
loctimes = list(zip(locs,times,rows,dcm_inds))
from operator import itemgetter
loctimes.sort(key=itemgetter(1))
loctimes.sort(key=itemgetter(0))
loctimes.sort(key=itemgetter(2))

sort_inds = [ele[3] for ele in loctimes]
# combine into array with sorting order
image_volume = np.stack([dcms[i].pixel_array for i in sort_inds[165:]])
multi_slice_viewer0(image_volume/1000)