#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:55:35 2018

@author: jmj136
"""

datapath = 'lowdosePETdata_v3.hdf5'
outpathLD = '/home/jmj136/deep-learning/PETimages/lowdose{:05d}.jpeg'
outpathFD = '/home/jmj136/deep-learning/PETimages/fulldose{:05d}.jpeg'
import h5py
import numpy as np

print('Loading data...')
with h5py.File(datapath,'r') as f:
    x_train = np.array(f.get('train_inputs'))
    y_train = np.array(f.get('train_targets'))

import scipy.misc
for ii in range(x_train.shape[0]):
    scipy.misc.toimage(x_train[ii,...,0], cmin=0.0, cmax=1.5).save(outpathLD.format(ii))
    scipy.misc.toimage(y_train[ii,...,0], cmin=0.0, cmax=1.5).save(outpathFD.format(ii))

#from PIL import Image
#im = Image.fromarray(x_train[ii,...,0]).convert('RGB')
#im.save(outpathLD.format(ii))
#im = Image.fromarray(y_train[ii,...,0]).convert('RGB')
#im.save(outpathFD.format(ii))
