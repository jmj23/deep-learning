# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import numpy as np
import glob
import scipy.io as spio
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
from keras.models import load_model
K._LEARNING_PHASE = tf.constant(0)

DataDir = 'MatData'
filepath = "{}/PETreconTrainingData{:03d}.mat".format(DataDir, 1)
outputdir = "MuMapOutput"
savepath = "{}/MuMapDataSubject{:03d}.mat".format(outputdir,1)
numsubj = 11
#%% Load model
with tf.device('/cpu:0'):
    RegModel = load_model("MuMapModel_v4.hdf5")
    
#%% Inputs
print('Loading inputs')
globfiles = sorted(glob.glob("MatData/*.mat"))
mat = spio.loadmat(globfiles[0],squeeze_me=True)
ims = mat['x']
ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
wims = ims[...,0]
fims = ims[...,1]
for im in wims:
    im /= np.max(im)
for im in fims:
    im /= np.max(im)

inputs = np.stack((wims,fims),axis=3)
numSlices = [inputs.shape[0]]

for file in globfiles[1:numsubj]:
    print(file[-7:-4])
    mat = spio.loadmat(file,squeeze_me=True)
    ims = mat['x']
    ims = np.rollaxis(ims.astype(np.float32, copy=False),2,0)
    wims = ims[...,0]
    fims = ims[...,1]
    for im in wims:
        im /= np.max(im)
    for im in fims:
        im /= np.max(im)
    
    new_inputs = np.stack((wims,fims),axis=3)
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    numSlices.append(new_inputs.shape[0])
    
inputs[inputs<0] = 0

#%% Make predictions on MR data
output = RegModel.predict(inputs,batch_size=16)
output[output<0] = 0
muMaps = output[...,0]

#%% Store to MAT files
for ss in range(numsubj):
    curMap = np.rollaxis(muMaps[:numSlices[ss],...],0,3)
    savepath = "{}/MuMapDataSubject{:03d}.mat".format(outputdir,ss+1)
    spio.savemat(savepath,mdict={'MuMap':curMap})
    muMaps = np.delete(muMaps,np.s_[0:numSlices[ss]],axis=0)

