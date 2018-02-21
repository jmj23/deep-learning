# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras.models import load_model
from matplotlib import pyplot as plt
from CustomMetrics import jac_met, dice_coef, dice_coef_loss, perc_error
import numpy as np
import h5py
import time

#%% load data
# Model Save Path/name
#model_filepath = 'ResSegModel_v9.hdf5'
#model_filepath = 'BlockSegModel_v1.hdf5'
model_filepath = 'BlockSegModel_v2.hdf5'
# Data path/name
val_datapath = 'segdata_ranked_val.hdf5'
with h5py.File(val_datapath,'r') as f:
    x_val = np.array(f.get('inputs'))
    y_val = np.array(f.get('targets'))
    

#%% load model
SegModel = load_model(model_filepath,
                          custom_objects={'jac_met':jac_met,
                                          'perc_error':perc_error,
                                          'dice_coef':dice_coef,
                                          'dice_coef_loss':dice_coef_loss})
#%% evaluate model

print('Evaluating model...')
scores = SegModel.evaluate(x_val,y_val,batch_size=16)
eval_scores = "Dice Loss: {0[0]:.4f}, Dice Coef: {0[1]:.4f}, Jaccard Coef: {0[2]:.4f}, Percent Error: {0[3]:.3f}%".format(scores)
print('')
print(eval_scores)


# get slice-by-slice scores
pr_bs = np.minimum(16,x_val.shape[0])
time1 = time.time()
output = SegModel.predict(x_val,batch_size=pr_bs)
time2 = time.time()
print('Infererence time: ',(time2-time1)/x_val.shape[0],' per slice')
test_masks = output[...,0]>.5
test_targs = y_val[...,0].astype(np.bool)
error = np.not_equal(test_masks,test_targs)
errorsum = np.sum(error,axis=(1,2))
denomsum = np.sum(test_targs,axis=(1,2))
PE = 100*errorsum/denomsum
fig1 = plt.figure()
plt.plot(PE,'r-')
plt.xlabel('Slice number')
plt.ylabel('Percent Error')

fig2 = plt.figure()
plt.hist(PE,bins=50,edgecolor='black',normed=True)
plt.xticks(np.arange(0,np.round(np.max(PE)), 2))
plt.title('Percent Error per Slice')
plt.xlabel('Percent Error')
plt.ylabel('Proportion of slices')


# display example masks
lim = np.minimum(x_val.shape[0],200)
print('Generating masks')
from VisTools import mask_viewer0
w_ims = x_val[...,0]
msks = output[...,0]
cor_msks = y_val[...,0]
mask_viewer0(w_ims[-lim:-1,...],msks[-lim:-1,...],'Sample Masks')
mask_viewer0(w_ims[-lim:-1,...],cor_msks[-lim:-1,...],'Correct Masks')
mask_viewer0(w_ims[-lim:-1,...],error[-lim:-1,...].astype(np.int),'Error Masks')

