#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:26:16 2017

@author: jmj136
"""

from keras.models import load_model
#import my_callbacks
from matplotlib import pyplot as plt
from CustomMetrics import jac_met, dice_coef, dice_coef_loss, perc_error
import numpy as np
import h5py
from VisTools import mask_viewer0

model_filepath = 'ResSegModel_v5.hdf5'
val_datapath = 'segdata_val_aug.hdf5'
with h5py.File(val_datapath,'r') as f:
    x_val = np.array(f.get('inputs'))
    y_val = np.array(f.get('targets'))


SegModel = load_model(model_filepath,
                      custom_objects={'jac_met':jac_met,
                                      'perc_error':perc_error,
                                      'dice_coef':dice_coef,
                                      'dice_coef_loss':dice_coef_loss})

val_output = SegModel.predict(x_val,batch_size=20)

w_ims = x_val[...,0]
seg_msks = (val_output[...,0]>.5).astype(np.float)
truth_msks = (y_val[...,0]>.5).astype(np.float)
f_ims = x_val[...,1]
PDWF = np.nan_to_num(np.divide(w_ims,w_ims+f_ims))
PDWF[PDWF>1]=1
PDWF[PDWF<0]=0

PDWFmasked = truth_msks*PDWF
vols = np.sum(truth_msks,axis=(1,2))
PDWFsums = np.sum(PDWFmasked,axis=(1,2))/vols

intersect = np.sum(seg_msks*truth_msks,axis=(1,2))
denom = np.sum(seg_msks,axis=(1,2))+np.sum(seg_msks,axis=(1,2))
dice = 2*intersect/denom

plt.scatter(PDWFsums, dice)
plt.show()

sortinds = np.argsort(dice)

sort_ims = np.take(w_ims,sortinds,axis=0)
sort_msks = np.take(seg_msks,sortinds,axis=0)

mask_viewer0(sort_ims,sort_msks)