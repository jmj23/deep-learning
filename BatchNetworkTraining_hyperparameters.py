# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras import optimizers
from CustomMetrics import dice_coef, dice_coef_loss, perc_error
import numpy as np
import h5py
import BatchModels
import time
import csv
import tensorflow as tf

np.random.seed(42)

#%% Load and Prepare Data

# Model Save Path/name
model_filepath = 'BatchModel'
# Data path/name
batch_datapath = 'batch_segdata.hdf5'

with h5py.File(batch_datapath,'r') as f:
        inputs = np.array(f.get('inputs'))
        targets = np.array(f.get('targets'))
    
# pull out validation set
val_rat = 0.30
np.random.seed(seed=1)
numS = inputs.shape[0]
numVal = np.round(val_rat*numS).astype(np.int)
valvec = np.random.choice(numS, numVal, replace=False)

val_inputs = np.take(inputs,valvec,axis=0)
inputs = np.delete(inputs,valvec,axis=0)
    
# randomize inputs
print('sorting inputs')
numS = inputs.shape[0]
sort_r = np.random.permutation(numS)
np.take(inputs,sort_r,axis=0,out=inputs);

# pull out validation set of targets
val_targets = np.take(targets,valvec,axis=0)
targets = np.delete(targets,valvec,axis=0)

# sort targets same as inputs
print('sorting targets')
np.take(targets,sort_r,axis=0,out=targets);

#%% Create csv results file
csvfile = "BatchTrainingResults_hyperparameters.csv"
csvfile_plots = "BatchTrainingData_hyperparameters.csv"

#Assuming res is a flat list
with open(csvfile, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Results'])
    writer.writerow(['Model Number','Model Description','Batch Size',
                     'Learning Rate','Percent Error','Dice Coefficient',
                     'PE Std Dev'])
with open(csvfile_plots, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Plot Data'])   
    
#%% Setup hyperparameter sets
BatchSizes = [2,4,8,16]
LearningRates = [10**x for x in [-1,-2,-3,-3]]

#%% Loop over all parameter sets
mnum = 27
ModelName = 'BatchModel_v{}'.format(mnum)
model_fcn = getattr(BatchModels, ModelName)
for BS in BatchSizes:
    for LR in LearningRates:
        print('Running param set')
        descrp = 'Batch Size: {}, Learning Rate: {:e}'.format(BS,LR)
        print(descrp)
        #%% prepare model for training
        tf.set_random_seed(42)
        
        print("Generating " + ModelName)
        [BatchModel,label] = model_fcn(inputs)
        sgd = optimizers.sgd(lr=LR, decay=1e-4*LR, momentum=0.9, nesterov=True)
        BatchModel.compile(loss=dice_coef_loss, optimizer=sgd,
                  metrics=[dice_coef, perc_error])
        
        #%% training
        print('Starting training')
        numEp = 40
        b_s = BS
        history = BatchModel.fit(inputs, targets,
                           batch_size=b_s, epochs=numEp,
                           validation_data=(val_inputs,val_targets),
                           verbose=0)
        print('Model {} Training complete'.format(mnum))
                
        #%% evaluate model
        print('Evaluating {}...'.format(ModelName))        
        
        # calculate per-slice percent error
        pr_bs = 16
        time1 = time.time()
        output = BatchModel.predict(val_inputs,batch_size=pr_bs)
        time2 = time.time()
        test_masks = output[...,0]>.5
        test_targs = val_targets[...,0].astype(np.bool)
        error = np.not_equal(test_masks,test_targs)
        errorsum = np.sum(error,axis=(1,2))
        denomsum = np.sum(test_targs,axis=(1,2))
        PE = errorsum/denomsum
        PEstd = np.std(PE)
        
        #%% save results to csv
        minPE = np.min(history.history['val_perc_error'])
        maxDSI = np.max(history.history['val_dice_coef'])
        res = [mnum,label,BS,LR] + [minPE,maxDSI,PEstd]
        res2 = ['Train'] + [label] + [BS,LR] + history.history['perc_error'] + ['Validation'] + [label] + history.history['val_perc_error']
        with open(csvfile, "a") as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(res) 
        with open(csvfile_plots, "a") as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(res2)
        print('Results saved')