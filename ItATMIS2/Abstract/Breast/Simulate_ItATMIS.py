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
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from ItATMISfunctions import BlockModel, dice_coef_loss
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
savepath = '/data/jmj136/ItATMIS/CAP/data.hdf5'
model_weights_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/CAP/best_model_weights.h5'
val_frac = 0.2
cross_val_num = 1

cb_check = ModelCheckpoint(model_weights_path,
                           monitor='val_loss',
                           verbose=0,save_best_only=True,
                           save_weights_only=True,
                           mode='auto',period=1)
cb_eStop = EarlyStopping(monitor='val_loss',patience=6,verbose=1,mode='auto')

CBs = [cb_check,cb_eStop]

#%% Load data
subjs = [0,1,2]
inputs = []
targets = []
for subj in subjs:
    with h5py.File(savepath) as hf:
        inputs.append(np.array(hf.get('subj_{:03d}_images'.format(subj))))
        targets.append(np.array(hf.get('subj{:03d}_masks'.format(subj))))

    
# make model
model = BlockModel(inputs.shape,filt_num=16,numBlocks=4,num_out_channels=1)
model.compile(optimizer=Adam, loss=dice_coef_loss)

# split off cross-validation subjects
cv_subjs = random.sample(subjs,cross_val_num)
cv_inputs = np.concatenate([i for j, i in enumerate(inputs) if j in cv_subjs])
cv_targets = np.concatenate([i for j, i in enumerate(inputs) if j in cv_subjs])
for i in cv_subjs:
    inputs.pop(i)
    targets.pop(i)
    
#%% Iteratively train and add subjects

#for cur_num_subj in range(1,len(inputs)+1):
for cur_num_subj in [1]:
    cur_inputs = np.concatenate(inputs[:cur_num_subj])
    cur_targets = np.concatenate(targets[:cur_num_subj])
    # split off validation data
    numIm = cur_inputs.shape[0]
    val_inds = np.random.choice(np.arange(numIm),
                                np.round(val_frac*numIm).astype(np.int),
                                replace=False)
    valX = np.take(inputs,val_inds,axis=0)
    valY = np.take(targets,val_inds, axis=0)
    trainX = np.delete(inputs, val_inds, axis=0)
    trainY = np.delete(targets, val_inds, axis=0)
    
    
    # setup image data generator
    datagen1 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    datagen2 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
     
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    datagen1.fit(trainX, seed=seed)
    datagen2.fit(trainY, seed=seed)
    batchsize = 16
    datagen = zip( datagen1.flow( trainX, None, batchsize, seed=seed), datagen2.flow( trainY, None, batchsize, seed=seed) )
    
    # calculate number of epochs and batches
    # numEp = np.maximum(40,np.minimum(np.int(10*(self.FNind+1)),100))
    numEp = 50
    steps = np.minimum(np.int(trainX.shape[0]/batchsize*16),1000)
    numSteps = steps*numEp
    
    model.fit_generator(datagen,
                        steps_per_epoch=steps,
                        epochs=numEp,
                        callbacks=CBs,
                        validation_data=(valX,valY))
    print('Training complete. Loading best weights')
    model.load_weights(model_weights_path)
    print('Evaluating model on cross-validation data')
    score = model.evaluate(cv_inputs,cv_targets)
    print("Cross-validation Dice score is {:.03f}".format(1-score))
    
    