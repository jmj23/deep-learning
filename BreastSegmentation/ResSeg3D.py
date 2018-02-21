# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils.io_utils import HDF5Matrix
from matplotlib import pyplot as plt
from CustomMetrics import jac_met, dice_coef, dice_coef_loss
from my_callbacks import Histories
import numpy as np
import h5py
import json
import scipy.io as spio
import Kmodels

# Model Save Path/name
model_filepath = 'ResSegModel_3D.hdf5'
# Data path/name
datapath = 'segdata_3D.hdf5'

#%% Loading data
#mat = spio.loadmat('Data/dataset25.mat',squeeze_me=True)
#x_train = mat['InputArray'].astype(np.float32)
#y_train = mat['TargetArray'][...,0].astype(np.float32)
#x_train = x_train.reshape(1,x_train.shape[0],256,256,2)
#y_train = y_train.reshape(1,y_train.shape[0],256,256,1)
#xr = [30,120]
#yr = [90,180]
#zr = [0,20]
#x_train= x_train[:,zr[0]:zr[1],yr[0]:yr[1],xr[0]:xr[1],:]
#y_train= y_train[:,zr[0]:zr[1],yr[0]:yr[1],xr[0]:xr[1],:]

#x_train = HDF5Matrix(datapath, 'inputs')
#y_train = HDF5Matrix(datapath, 'targets')
#
with h5py.File(datapath,'r') as f:
    numsamp = 1
    inputs = f.get('inputs')
    x_train = np.array(inputs[:numsamp,:,:,:,:])
    targets = f.get('targets')
    y_train = np.array(targets[:numsamp,:,:,:,:])

#%% Setting callbacks
earlyStopping = EarlyStopping(monitor='val_loss',patience=6, verbose=0,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

hist = Histories()

#CBs = [checkpoint,earlyStopping,hist]
CBs = [hist]



#%% prepare model for training
redo = 0
if redo==0:
    print("Generating new model")
    SegModel3D = Kmodels.ResModel_3D_elu_small(x_train)
    
    SegModel3D.compile(loss=dice_coef_loss, optimizer='adam',
                  metrics=[dice_coef, jac_met])
else:
    print('Reloading previous model')
    SegModel3D = load_model(model_filepath,
                          custom_objects={'jac_met':jac_met,'dice_coef':dice_coef,
                                          'dice_coef_loss':dice_coef_loss})


#%% training
print('Starting training')
numEp = 5
b_s = 1
v_s = 0
history = SegModel3D.fit(x_train, y_train,
                       batch_size=b_s, epochs=numEp,
                       validation_split=v_s,
                       verbose=1,
                       callbacks=CBs)

print('Training complete. Plotting indices')

#%% plotting loss
step = b_s/x_train.shape[0]
actEpochs = len(history.history['loss'])
epochs = np.arange(1,actEpochs+1)
actBatches = len(hist.loss)
batches = np.arange(1,actBatches+1)* actEpochs/(actBatches+1)
fig2 = plt.figure();
plt.plot(epochs,history.history['loss'],'r-s')
plt.plot(batches,hist.loss,'r-')

plt.plot(epochs,history.history['dice_coef'],'b-s')
plt.plot(batches,hist.dice,'b-')

plt.plot(epochs,history.history['jac_met'],'g-s')
plt.plot(batches,hist.jac,'g-')

plt.plot(epochs,history.history['val_loss'],'o-s')
plt.plot(batches,hist.val_loss,'o-')

plt.plot(epochs,history.history['val_dice_coef'],'c-s')
plt.plot(batches,hist.val_dice,'c-')

plt.plot(epochs,history.history['val_jac_met'],'y-s')
plt.plot(batches,hist.val_jac,'y-')

plt.title('Model Metrics')
plt.ylabel('Index')
plt.xlabel('epoch')

#plt.legend(['dice loss', 'dice metric', 'jacc metric'], loc='upper left')
plt.legend(['train loss', 'train loss',
            'train DSI', 'train DSI',
            'train JSI', 'train JSI',
            'val loss', 'val loss'
            'val DSI', 'val DSI',
            'val JSI', 'val JSI'],
            loc='upper left')
plt.show()

#%% Plotting results

print('Generating sample plot')
# segmentation result
output = SegModel3D.predict(x_train,batch_size=b_s)

from VisTools import mask_viewer0
w_ims = x_train[0,:,:,:,0]
msks = output[0,:,:,:,0]
mask_viewer0(w_ims,msks)

#%% evaluate model
print('Evaluating model...')
scores = SegModel3D.evaluate(x_train,y_train,batch_size=b_s)
eval_scores = "Dice Loss: {0[0]:.4f}, Dice Coef: {0[1]:.4f}, Jaccard Coef: {0[2]:.4f}".format(scores)
print(eval_scores)

#%% save model architecture
#model_arch = SegModel.to_json()
#with open(model_filepath[:-5]+'_ModelArch.txt','w') as file:
#    json.dump(model_arch,file)
#    
## save model weights
#SegModel.save_weights(model_filepath[:-5]+'_weights.h5')