# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import optimizers
#import my_callbacks
from matplotlib import pyplot as plt
from CustomMetrics import jac_met, dice_coef, dice_coef_loss, perc_error
from my_callbacks import Histories, Plots
import numpy as np
import h5py
import json
import Kmodels
import time

#%%
# Model Save Path/name
model_filepath = 'ResSegModel_v9.hdf5'
# Data path/name
train_datapath = 'segdata_ranked_train.hdf5'
val_datapath = 'segdata_ranked_val.hdf5'
with h5py.File(train_datapath,'r') as f:
    inputs = f.get('inputs')
    x_train = np.array(inputs)
    targets = f.get('targets')
    y_train = np.array(targets)
with h5py.File(val_datapath,'r') as f:
    x_val = np.array(f.get('inputs'))
    y_val = np.array(f.get('targets'))

#%% callbacks
earlyStopping = EarlyStopping(monitor='val_loss',patience=10, verbose=1,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

hist = Histories()

CBs = [checkpoint,earlyStopping,hist]

#%% prepare model for training
redo = 0
if redo==0:
    print("Generating new model")
    SegModel = Kmodels.ResModel_v9(x_train)
    adopt = optimizers.adadelta()
    SegModel.compile(loss=dice_coef_loss, optimizer=adopt,
                  metrics=[dice_coef, jac_met, perc_error])
else:
    print('Reloading previous model')
    SegModel = load_model(model_filepath,
                          custom_objects={'jac_met':jac_met,
                                          'perc_error':perc_error,
                                          'dice_coef':dice_coef,
                                          'dice_coef_loss':dice_coef_loss})

#%% training
print('Starting training')
numEp = 100
b_s = 8
history = SegModel.fit(x_train, y_train,
                   batch_size=b_s, epochs=numEp,
                   validation_data=(x_val,y_val),
                   verbose=1,
                   callbacks=CBs)

print('Training complete')

#%% evaluate model

if True:
    # load checkpoint model
    SegModel = load_model(model_filepath,
                          custom_objects={'jac_met':jac_met,
                                          'perc_error':perc_error,
                                          'dice_coef':dice_coef,
                                          'dice_coef_loss':dice_coef_loss})   
    # save model architecture 
    model_arch = SegModel.to_json()
    with open(model_filepath[:-5]+'_ModelArch.txt','w') as file:
        json.dump(model_arch,file)
        
    # save model weights
    SegModel.save_weights(model_filepath[:-5]+'_weights.h5')
    print('Model architecture and weights saved to ' + model_filepath[:-5]+ '_weights')
    
print('Evaluating model...')
time1 = time.time()
scores = SegModel.evaluate(x_val,y_val,batch_size=16)
time2 = time.time()
eval_scores = "Dice Loss: {0[0]:.4f}, Dice Coef: {0[1]:.4f}, Jaccard Coef: {0[2]:.4f}, Percent Error: {0[3]:.3f}%".format(scores)
print(eval_scores)
print('Infererence time: ',(time2-time1)/x_val.shape[0],' per slice')

#%% plotting
print('Plotting metrics')
step = np.minimum(b_s/x_train.shape[0],1)
actEpochs = len(history.history['loss'])
epochs = np.arange(1,actEpochs+1)
actBatches = len(hist.loss)
batches = np.arange(1,actBatches+1)* actEpochs/(actBatches+1)

fig2 = plt.figure(1,figsize=(12.0, 6.0));
plt.plot(epochs,history.history['loss'],'r-s')
plt.plot(batches,hist.loss,'r-')

plt.plot(epochs,history.history['dice_coef'],'b-s')
plt.plot(batches,hist.dice,'b-')

plt.plot(epochs,history.history['jac_met'],'g-s')
plt.plot(batches,hist.jac,'g-')
try:
    plt.plot(epochs,history.history['val_loss'],'m-s')
    plt.plot(batches,hist.val_loss,'m-')
    
    plt.plot(epochs,history.history['val_dice_coef'],'c-s')
    plt.plot(batches,hist.val_dice,'c-')
    
    plt.plot(epochs,history.history['val_jac_met'],'y-s')
    plt.plot(batches,hist.val_jac,'y-')
except Exception:
    pass
plt.title('Model Metrics')
plt.ylabel('Index')
plt.xlabel('epoch')
if len(x_val)==0:
    plt.legend(['dice loss', 'dice metric', 'jacc metric'], loc='upper left')
else:
    plt.legend(['train loss', 'train loss',
                'train DSI', 'train DSI',
                'train JSI', 'train JSI',
                'val loss', 'val loss',
                'val DSI', 'val DSI',
                'val JSI', 'val JSI'],
                loc='upper left')
plt.show()
plt.savefig("SavedPlots/{}_TrainMetrics.png".format(model_filepath),dpi = 200)

if len(x_val)==0:
    lim = np.minimum(x_train.shape[0],200)
    print('Generating masks')
    # segmentation result
    pr_bs = np.minimum(10,x_train.shape[0])
    output = SegModel.predict(x_train[0:lim,:,:,:],batch_size=pr_bs)
    
    from VisTools import mask_viewer0
    w_ims = x_train[0:lim,:,:,0]
    msks = output[:,:,:,0]
    mask_viewer0(w_ims,msks)
else:
    lim = np.minimum(x_val.shape[0],200)
    print('Generating masks')
    # segmentation result
    pr_bs = np.minimum(16,x_val.shape[0])
    output = SegModel.predict(x_val[-lim:-1,:,:,:],batch_size=pr_bs)
    
    from VisTools import mask_viewer0
    w_ims = x_val[-lim:-1,:,:,0]
    msks = output[:,:,:,0]
    mask_viewer0(w_ims,msks)

if False:
    from VisTools import save_masked_image
    w_ims = x_val[0:lim,:,:,0]
    msks = output[...,0]
    save_masked_image(w_ims,msks,name='SegOutput')