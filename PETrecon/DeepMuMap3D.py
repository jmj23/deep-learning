# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
sys.path.insert(0,'/home/jmj136/KerasFiles')
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.models import load_model
from matplotlib import pyplot as plt
from my_callbacks import Histories
import numpy as np
import h5py
import time

import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#%%
# Model Save Path/name
model_filepath = 'MuMapModel3D_v{}.hdf5'.format(1)
# Data path/name
datapath = 'petrecondata3D_tvt.hdf5'

with h5py.File(datapath,'r') as f:
    x_train = np.array(f.get('train_inputs'))
    y_train = np.array(f.get('train_targets'))
    x_val = np.array(f.get('val_inputs'))
    y_val = np.array(f.get('val_targets'))
    x_test = np.array(f.get('test_inputs'))
    y_test = np.array(f.get('test_targets'))

#%% callbacks
earlyStopping = EarlyStopping(monitor='val_loss',patience=16,verbose=1,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

hist = Histories()

CBs = [checkpoint,earlyStopping,hist]

#%% prepare model for training
print("Generating model")
from RegressionModels import BlockModel3D
RegModel = BlockModel3D(x_train.shape[1:],12,3)

adopt = optimizers.adam(clipnorm=1)
sgd = optimizers.SGD(lr=1e-6,momentum=0.9,decay=1e-3,nesterov=True,clipnorm=1)
RegModel.compile(loss='MSE', optimizer=adopt)

#%% training
print('Starting training')
numEp = 50
b_s = 1
history = RegModel.fit(x_train, y_train,
                   batch_size=b_s, epochs=numEp,
                   validation_data=(x_val,y_val),
                   verbose=1,
                   callbacks=CBs)

print('Training complete')

print('Loading best model...')
RegModel = load_model(model_filepath)

score = RegModel.evaluate(x_test,y_test)
print("")
print("MSE on test data: {}".format(score))

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
plt.plot(epochs,history.history['val_loss'],'m-s')
plt.plot(batches,hist.val_loss,'m-')

plt.show()

print('Generating samples')
# regression result
pr_bs = np.minimum(2,x_test.shape[0])
time1 = time.time()
output = RegModel.predict(x_test,batch_size=pr_bs)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/x_test.shape[0],' ms per slice')

from skimage.measure import compare_ssim as ssim
SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_test[...,0],output[...,0])]

num_bins = 10
fig3 = plt.figure()
n, bins, patches = plt.hist(SSIMs, num_bins, facecolor='blue', edgecolor='black', alpha=0.5)
plt.show()
print('Mean SSIM of ', np.mean(SSIMs))
print('SSIM range of ', np.round(np.min(SSIMs),3), ' - ', np.round(np.max(SSIMs),3))

from VisTools import multi_slice_viewer0
multi_slice_viewer0(np.c_[x_test[0,...,0],output[0,...,0],y_test[0,...,0]],[])
