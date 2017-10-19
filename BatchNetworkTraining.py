# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras import optimizers
#import my_callbacks
from matplotlib import pyplot as plt
from CustomMetrics import dice_coef, dice_coef_loss, perc_error
import numpy as np
import h5py
import BatchModels
import time
import csv
#from VisTools import mask_viewer0

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
csvfile = "BatchTrainingResults.csv"
csvfile_plots = "BatchTrainingData.csv"

#Assuming res is a flat list
with open(csvfile, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Results'])
    writer.writerow(['Model Number','Model Description','Percent Error','Dice Coefficient',
                     'Number of Parameters','Inference Time','Training Time'])
with open(csvfile_plots, "w") as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['Plot Data'])   

#%% Loop over all models
for mnum in [8,13,14,15,16,17,18,19,20,22,23]:
    #%% prepare model for training
    ModelName = 'BatchModel_v{}'.format(mnum)
    
    model_fcn = getattr(BatchModels, ModelName)
    
    print("Generating " + ModelName)
    [BatchModel,label] = model_fcn(inputs)
    adopt = optimizers.adadelta()
    output_shape = BatchModel.compute_output_shape(inputs.shape)
    if len(output_shape) == 2:
        BatchModel.compile(loss=dice_coef_loss, loss_weights=[1., 0.2],
                           optimizer=adopt, metrics=[dice_coef, perc_error])
    else:
        BatchModel.compile(loss=dice_coef_loss, optimizer=adopt,
                  metrics=[dice_coef, perc_error])
    
    #%% training
    print('Starting training')
    numEp = 50
    b_s = 8
    if len(output_shape) == 2:
        time1 = time.time()
        history = BatchModel.fit(inputs, [targets,targets],
                           batch_size=b_s, epochs=numEp,
                           validation_data=(val_inputs,[val_targets,val_targets]),
                           verbose=0)
        time2 = time.time()
    else:
        time1 = time.time()
        history = BatchModel.fit(inputs, targets,
                           batch_size=b_s, epochs=numEp,
                           validation_data=(val_inputs,val_targets),
                           verbose=0)
        time2 = time.time()
    training_time = time2-time1
    print('Model {} Training complete'.format(mnum))
    print('Training time: {:.2f} min'.format(training_time/60))
    
    #%% plotting percent error
    print('Plotting metrics')
    actEpochs = len(history.history['loss'])
    epochs = np.arange(1,actEpochs+1)
    
    fig1 = plt.figure(figsize=(12.0, 6.0));
    
    plt.plot(epochs,history.history['perc_error'],'r-s')
    plt.plot(epochs,history.history['val_perc_error'],'g-s')
    
    plt.title(ModelName + ' Metrics')
    plt.ylabel('Percent Error')
    plt.xlabel('epoch')
    
    plt.legend(['train percent error',
                'val percent error'],
                loc='upper left')
    plt.show()
#    plt.savefig("SavedPlots/{}_PercentErrorPlot.png".format(ModelName),dpi = 200)
    plt.close(fig1)
    
    #%% evaluate model
    print('Evaluating {}...'.format(ModelName))
    
    scores = BatchModel.evaluate(val_inputs,val_targets,batch_size=16,verbose=0)
    
    eval_scores = "Dice Loss: {0[0]:.4f}, Dice Coef: {0[1]:.4f}, Percent Error: {0[2]:.3f}%".format(scores)
    print(eval_scores)
    
    
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
    fig2 = plt.figure()
    plt.hist(100*PE,bins=25,edgecolor='black')
    plt.xticks(np.arange(0,105, 5))
    plt.title(ModelName + ' Percent Error per Slice')
    plt.xlabel('Percent Error')
    plt.ylabel('Number of slices')
    plt.savefig("SavedPlots/{}_PercentErrorHistogram.png".format(ModelName),dpi = 200)
    plt.close(fig2)
    
    inf_time = (time2-time1)/val_inputs.shape[0]
    print('Infererence time for Model v{}: {:.4e} per slice'.format(mnum,inf_time))
    
    #%% save results to csv
    minPE = np.min(history.history['val_perc_error'])
    maxDSI = np.max(history.history['val_dice_coef'])
    res = [mnum] + [label] + [minPE,maxDSI] + [PEstd] + [BatchModel.count_params()] + [inf_time] + [training_time]
    res2 = ['Train'] + [label] + history.history['perc_error'] + ['Validation'] + [label] + history.history['val_perc_error']
    with open(csvfile, "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(res) 
    with open(csvfile_plots, "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(res2)
    print('Results saved')
    #%% see some results
#    print('Generating masks...')
#    # segmentation result
#    lim = 48        
#    w_ims = val_inputs[0:lim,...,0]
#    disp_masks = output[0:lim,...,0]>.5
#    disp_targs = val_targets[0:lim,...,0].astype(np.bool)
#    discrep = np.not_equal(test_masks,test_targs)
#    mask_viewer0(w_ims,discrep,name=ModelName)
#    
#    print('')
    
exit