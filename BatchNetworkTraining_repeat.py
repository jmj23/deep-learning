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
csvfile = "BatchTrainingResults_repeat.csv"
csvfile_plots = "BatchTrainingData_repeat.csv"

#Assuming res is a flat list
with open(csvfile, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Results'])
    writer.writerow(['Model Number','Repetition Number','Model Description',
                     'Percent Error','Dice Coefficient','PE Std Dev'])
with open(csvfile_plots, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Plot Data'])   

#%% Loop over all models
num_rep = 4
for mnum in [26,27]:
    for rep in range(num_rep):
        print('Running rep ',rep,'for model ',mnum)
        #%% prepare model for training
        ModelName = 'BatchModel_v{}'.format(mnum)
        
        model_fcn = getattr(BatchModels, ModelName)
        
        print("Generating " + ModelName)
        [BatchModel,label] = model_fcn(inputs)
        adopt = optimizers.adadelta()
        BatchModel.compile(loss=dice_coef_loss, optimizer=adopt,
                  metrics=[dice_coef, perc_error])
        
        #%% training
        print('Starting training')
        numEp = 50
        b_s = 8
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
        res = [mnum,rep,label] + [minPE,maxDSI,PEstd]
        res2 = ['Train'] + [label] + [rep] + history.history['perc_error'] + ['Validation'] + [label] + history.history['val_perc_error']
        with open(csvfile, "a") as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(res) 
        with open(csvfile_plots, "a") as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(res2)
        print('Results saved')