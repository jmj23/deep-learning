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
np.random.seed(seed=1)
from glob import glob
from time import time
from keras.callbacks import ModelCheckpoint,EarlyStopping
from ItATMISfunctions import BlockModel, dice_coef_loss, GetLCTSCdata
from ItATMISfunctions import SimulateItATMIS
from keras.optimizers import Adam
import random
random.seed(1)
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
    
import warnings
warnings.filterwarnings("ignore")
#%% Parameters
data_dir = '/data/jmj136/ItATMIS/LCTSC'
model_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/LCTSC/best_model_weights2.h5'
val_frac = 0.2
num_CV_folds = 30
train_groups = [5,10,15,20]

cb_eStop = EarlyStopping(monitor='val_loss',patience=3,verbose=1,mode='auto')
#%% Load data
# Now we'll get all the subject directories using glob
subj_dirs = glob(os.path.join(data_dir,'LCTSC*'))
# and feed those directories into another function that loads
# the dicoms and masks for each
data = [GetLCTSCdata(d) for d in subj_dirs]
# get all images together as inputs
inputs = [d[0] for d in data]
# get all masks together as targets
targets = [d[1] for d in data]
del data

    
#%% Split into cross validation folds
import more_itertools as mit
input_folds = [list(c) for c in mit.divide(num_CV_folds, inputs)]
target_folds = [list(c) for c in mit.divide(num_CV_folds, targets)]

#%% Iterate for each cross-validation
for it in range(num_CV_folds):
    # split off cross-validation subjects    
    cv_inputs = np.concatenate(input_folds[it])
    cv_targets = np.concatenate(target_folds[it])[...,np.newaxis]
    
    # join together rest of subjects    
    train_inputs = [i for j, i in enumerate(input_folds) if j not in [it]]
    train_inputs = [j for i in train_inputs for j in i]
    train_targets = [i for j, i in enumerate(target_folds) if j not in [it]]
    train_targets = [j for i in train_targets for j in i]
    
    combined = list(zip(train_inputs, train_targets))
    random.shuffle(combined)
    train_inputs[:], train_targets[:] = zip(*combined)
    #%% Iteratively train and add subjects
    
    # list for collecting losses
    CV_losses = []
    for cur_num_subj in train_groups:
        # concatenate current data
        cur_inputs = np.concatenate(train_inputs[:cur_num_subj])
        cur_targets = np.concatenate(train_targets[:cur_num_subj])[...,np.newaxis]
        # make model
        model = BlockModel(inputs[0].shape,filt_num=16,numBlocks=4,num_out_channels=1)
        model.compile(optimizer=Adam(), loss=dice_coef_loss)
        # make callbacks
        cb_check = ModelCheckpoint(model_weights_path,monitor='val_loss',
                                   verbose=0,save_best_only=True,
                                   save_weights_only=True,mode='auto',period=1)
        
        CBs = [cb_check,cb_eStop]
        print('---------------------------------------')
        print('Including {} subjects on CV fold {}...'.format(cur_num_subj,it+1))
        time1 = time()
        model = SimulateItATMIS(model,cur_inputs,cur_targets,CBs,val_frac)
        time2 = time()
        print('Training complete')
        print('Time elapsed: {:0.2f} s'.format(time2-time1))
        print('Evaluating model on cross-validation data')
        model.load_weights(model_weights_path)
        score = model.evaluate(cv_inputs,cv_targets,verbose=0)
        print("Cross-validation Dice score is {:.03f}".format(1-score))
        CV_losses.append(1-score)
    
    # save results
    print('Cross validation fold complete. Saving results...')
    results_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV{}.txt'.format('Lung',it+1)
    np.savetxt(results_path,np.array(CV_losses))

# plot score results
print('All cross-validation folds complete!')
