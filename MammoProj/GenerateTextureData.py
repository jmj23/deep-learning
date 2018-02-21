# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:11:37 2017

@author: jmj136
"""

import NoiseGenerator
import numpy as np
import cv2
import h5py
from GenerateTransformationMatrix import makeM
import time

np.random.seed(seed=1)
imageSize = 128
numInputs = 5000
lesionSize = .02
filename = 'texture_training_data_128.hdf5'
#%% generate inputs
inputs = np.zeros((numInputs,imageSize,imageSize))
comp_inputs = np.copy(inputs)
lesion_inputs = np.copy(inputs)
params = np.abs(np.random.normal(8,2,numInputs))
time1 = time.time()
for pp in range(0,numInputs):
    if np.mod(pp,10)==0:
        print('Generating texture ',pp,'/',numInputs)
    noise = NoiseGenerator.NoiseUtils(imageSize)
    noise.makeTexture(texture = noise.cloud,param=params[pp])
    for i in range(0,imageSize):
        for j in range(0,imageSize):
            inputs[pp,i,j] = noise.img[i,j]/255
time2 = time.time()
print('{:0.3f} s per texture'.format((time2-time1)/numInputs))
#%% Intialize distortions and lesions

# make transformation matrices
Marray = np.zeros((numInputs,3,3))
theta_z = np.random.normal(0,np.pi/48,numInputs)
tx = np.random.normal(0,imageSize/300,numInputs)
ty = np.random.normal(0,imageSize/300,numInputs)
sk_x = np.random.normal(0,1e-5,numInputs)
sk_y = np.random.normal(0,1e-5,numInputs)
sc_x = np.random.normal(1,.01/3,numInputs)
sc_y = np.random.normal(1,.01/3,numInputs)

for mm in range(0,numInputs):
    Marray[mm,...] = makeM(theta_z[mm],tx[mm],ty[mm],sk_x[mm],sk_y[mm],
                              sc_x[mm],sc_y[mm],imageSize)
# random vector of whether lesion will be added   
positives = np.random.binomial(1, .5, size=numInputs)==1
false_positives = np.random.binomial(1, .2, size=numInputs)==1

# random lesion locations
lb = .1*imageSize
ub = .9*imageSize
numLesions = np.sum(positives)+np.sum(false_positives)
loc_x = np.random.randint(lb,ub,numLesions)
loc_y = np.random.randint(lb,ub,numLesions)

# create simulated lesion
lesion_size = np.round(lesionSize*imageSize).astype(np.int)
y,x = np.ogrid[-lesion_size: lesion_size+1, -lesion_size: lesion_size+1]
lesion_blob = x**2+y**2 <= lesion_size**2+.5*lesion_size

#%% add distortions and lesions
ll = 0
for cc in range(0,numInputs):
    lesion_inputs[cc,...] = np.copy(inputs[cc,...])
    if false_positives[cc]:
        lesion_mask = np.zeros((imageSize,imageSize),dtype=bool)
        lesion_mask[loc_y[ll]-lesion_size:loc_y[ll]+lesion_size+1,
                    loc_x[ll]-lesion_size:loc_x[ll]+lesion_size+1] = lesion_blob
        inputs[cc,lesion_mask] = np.max(inputs[cc,...])
        lesion_inputs[cc,lesion_mask] = np.max(lesion_inputs[cc,...])
        ll += 1
    if positives[cc]:
        lesion_mask = np.zeros((imageSize,imageSize),dtype=bool)
        lesion_mask[loc_y[ll]-lesion_size:loc_y[ll]+lesion_size+1,
                    loc_x[ll]-lesion_size:loc_x[ll]+lesion_size+1] = lesion_blob
        lesion_inputs[cc,lesion_mask] = np.max(lesion_inputs[cc,...])
        ll += 1
    comp_inputs[cc,...] = cv2.warpPerspective(lesion_inputs[cc,...],Marray[cc,...],
                           (imageSize,imageSize),borderMode=0)

#%% store to HDF5 file
print('storing as HDF5')
with h5py.File(filename, 'w') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
with h5py.File(filename, 'a') as hf:
    hf.create_dataset("comp_inputs",  data=comp_inputs,dtype='f')
with h5py.File(filename, 'a') as hf:
    hf.create_dataset("lesion_inputs",  data=lesion_inputs,dtype='f')
with h5py.File(filename, 'a') as hf:
    hf.create_dataset("labels",  data=positives,dtype='i')
with h5py.File(filename, 'a') as hf:
    hf.create_dataset("false_positives",  data=false_positives,dtype='i')    
print('Done')
del inputs
del comp_inputs
del lesion_inputs
del positives