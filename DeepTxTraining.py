# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from matplotlib import pyplot as plt
from my_callbacks import Histories
import numpy as np
import Kmodels
import glob
import nibabel
import os
import random

#%%
# Model Save Path/name
model_filepath = 'DeepTxModel.hdf5'
data_folder = '/home/axm3/data/deepTx/inputs/train'
img_rows = 128
img_cols = 128

img1_files = glob.glob(os.path.join(data_folder,'T1_2mm_MNI_pdt*.gz'))
img_count = 0

def resliceToAxial( data ):
    return np.transpose( data, (2,0,1) )

x_train = np.empty(0)
y_train = np.empty(0)

for curr_img1_file in img1_files:
    img_count += 1
    
    # read in T1 image first
    curr_img1_nii = nibabel.load(curr_img1_file)
    
    print( curr_img1_file )
    
    # get filename of matching Tx image
    subj_id_suffix = curr_img1_file.split('pdt')
    curr_img2_file = os.path.join(data_folder,'Tx_MNI_pdt' + subj_id_suffix[1])
    curr_img2_nii = nibabel.load(curr_img2_file)
    
    # data reslicing
    curr_img1 = resliceToAxial(curr_img1_nii.get_data())
    curr_img2 = resliceToAxial(curr_img2_nii.get_data())
    
    # data normalization for T1 input (Tx images are normalized later, below)
    curr_img1 = curr_img1.astype('float32')
    mean_img1 = np.mean(curr_img1)
    std_img1 = np.std(curr_img1)
    curr_img1 -= mean_img1
    curr_img1 /= std_img1
        
    img1 = curr_img1[...,np.newaxis]
    img2 = curr_img2[...,np.newaxis]
    
    # data centering for Tx
    img2 = img2.astype('float32')
    mean_img2 = np.mean(img2)
    std_img2 = np.std(img2)
    img2 -= mean_img2
    img2 /= std_img2
    if x_train.size==0:
        x_train = img1
        y_train = img2
    else:
        x_train = np.concatenate((x_train,img1))
        y_train = np.concatenate((y_train,img2))
        
# split off validation data
val_split = .2
val_num = np.round(x_train.shape[0]*val_split).astype(np.int)
val_inds = random.sample(range(x_train.shape[0]),val_num)

x_val = np.take(x_train,val_inds,axis=0)
y_val = np.take(y_train,val_inds,axis=0)

x_train = np.delete(x_train,val_inds,axis=0)
y_train = np.delete(y_train,val_inds,axis=0)
    
#%% callbacks
earlyStopping = EarlyStopping(monitor='val_loss',patience=10, verbose=1,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

hist = Histories()

CBs = [checkpoint,earlyStopping,hist]
CBs = [hist]

#%% prepare model for training
print("Generating model")
RegModel = Kmodels.BlockModel_reg(x_train)
adopt = optimizers.adadelta()
RegModel.compile(loss='MSE', optimizer=adopt)

#%% training
print('Starting training')
numEp = 30
b_s = 8
history = RegModel.fit(x_train, y_train,
                   batch_size=b_s, epochs=numEp,
                   validation_data=(x_val,y_val),
                   verbose=1,
                   callbacks=CBs)

print('Training complete')

score = RegModel.evaluate(x_val,y_val)
print("")
print("MSE on validation data: {}".format(score))

#%% plotting
print('Plotting metrics')
step = np.minimum(b_s/x_train.shape[0],1)
actEpochs = len(history.history['loss'])
epochs = np.arange(1,actEpochs+1)
actBatches = len(hist.loss)
batches = np.arange(1,actBatches+1)* actEpochs/(actBatches+1)

fig2 = plt.figure(1,figsize=(12.0, 6.0));
plt.plot(batches,hist.loss,'r-')
plt.plot(epochs,history.history['val_loss'],'b-s')

plt.show()

print('Generating samples')
# regression result
pr_bs = np.minimum(16,x_train.shape[0])
output = RegModel.predict(x_train,batch_size=pr_bs)

from VisTools import multi_slice_viewer0
disp_inds = np.array(range(0,128*3,3))
ims = np.take(x_train[...,0],disp_inds,axis=0)
outs = np.take(output[...,0],disp_inds,axis=0)
corrs = np.take(y_train[...,0],disp_inds,axis=0)
multi_slice_viewer0(np.concatenate((ims,outs,corrs,np.abs(corrs-outs)),axis=2),[])