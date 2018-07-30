#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:36:41 2018

@author: jmj136
"""
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
sys.path.insert(1,'/home/jmj136/deep-learning/ItATMIS2')
import numpy as np
from glob import glob
from scipy import io
from time import time
from natsort import natsorted
from keras.callbacks import ModelCheckpoint,EarlyStopping
from ItATMISfunctions import BlockModel, dice_coef_loss, SimulateItATMIS, PlotResults
from keras.optimizers import Adam
import random
# Use first available GPU
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
#%% Parameters
data_dir = '/data/jmj136/ItATMIS/Breast'
model_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Breast/best_model_weights.h5'
val_frac = 0.2
cross_val_num = 1
maxEpochs = 5

cb_check = ModelCheckpoint(model_weights_path,
                           monitor='val_loss',
                           verbose=0,save_best_only=True,
                           save_weights_only=True,
                           mode='auto',period=1)
cb_eStop = EarlyStopping(monitor='val_loss',patience=3,verbose=1,mode='auto')

CBs = [cb_check,cb_eStop]

#%% Load data
files = natsorted(glob(os.path.join(data_dir,'*.mat')))
subjs = range(len(files))
inputs = []
targets = []
for subj in subjs:
    mat = io.loadmat(files[subj])    
    inputs.append(mat['InputArray'])
    targets.append(mat['TargetArray'][...,0])

#%% Iterate for each cross-validation
it = 0
# make model
model = BlockModel(inputs[0].shape,filt_num=16,numBlocks=4,num_out_channels=1)
model.compile(optimizer=Adam(), loss=dice_coef_loss)

# split off cross-validation subjects
cv_subjs = random.sample(subjs,cross_val_num)
cv_inputs = np.concatenate([i for j, i in enumerate(inputs) if j in cv_subjs])
cv_targets = np.concatenate([i for j, i in enumerate(targets) if j in cv_subjs])[...,np.newaxis]
train_inputs = [i for j, i in enumerate(inputs) if j not in cv_subjs]
train_targets = [i for j, i in enumerate(targets) if j not in cv_subjs]

combined = list(zip(train_inputs, train_targets))
random.shuffle(combined)
train_inputs[:], train_targets[:] = zip(*combined)
it += 1
#%% Iteratively train and add subjects

# list for collecting losses
CV_losses = []
# maximim iterations
maxIter = np.minimum(len(train_inputs),maxEpochs)
for cur_num_subj in range(1,maxIter+1):
    cur_inputs = np.concatenate(train_inputs[:cur_num_subj])
    cur_targets = np.concatenate(train_targets[:cur_num_subj])[...,np.newaxis]
    print('Including {} subjects on CV fold {}'.format(cur_num_subj,it))
    time1 = time()
    model = SimulateItATMIS(model,cur_inputs,cur_targets,CBs,val_frac)
    time2 = time()
    print('Training complete')
    print('Time elapsed: {:02d} s'.format(time2-time1))
    print('Evaluating model on cross-validation data')
    model.load_weights(model_weights_path)
    score = model.evaluate(cv_inputs,cv_targets)
    print("Cross-validation Dice score is {:.03f}".format(1-score))
    CV_losses.append(1-score)

# save results
print('Cross validation fold complete. Saving results...')
results_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV{}'.format('breast',''.join(map(str,cv_subjs)))
np.savetxt(results_path,np.array(CV_losses))

# plot score results
print('All cross-validation folds complete!')
print('Displaying results')
PlotResults('breast')