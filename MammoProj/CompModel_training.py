# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras import optimizers
from matplotlib import pyplot as plt
import numpy as np
import h5py
import CompModels
import Kvis

#%% Loading Data
# Model Save Path/name
model_filepath = 'CompModel_v2.hdf5'
# Data path/name
datapath = 'texture_training_data_128.hdf5'
with h5py.File(datapath,'r') as f:
    inputs = np.array(f.get('inputs'))
    comp_inputs = np.array(f.get('comp_inputs'))
    lesion_inputs = np.array(f.get('lesion_inputs'))
    labels = np.array(f.get('labels'),dtype=np.int8)
    fps = np.array(f.get('false_positives'),dtype=np.int8)
inputs = inputs[...,np.newaxis]/np.max(inputs)
# decide if 2nd inputs are misregistered or not
dataset=1
if dataset==0:
    comp_inputs = comp_inputs[...,np.newaxis]/np.max(comp_inputs)
elif dataset==1:
    comp_inputs = lesion_inputs[...,np.newaxis]/np.max(lesion_inputs)

# take out validation set
numVal = np.round(inputs.shape[0]*.2).astype(np.int)
val_vec = np.random.choice(inputs.shape[0],numVal, replace=False)
x_val1 = np.take(inputs,val_vec,axis=0)
x_val2 = np.take(comp_inputs,val_vec,axis=0)
y_val = np.take(labels,val_vec)
x_train1 = np.delete(inputs,val_vec,axis=0)
x_train2 = np.delete(comp_inputs,val_vec,axis=0)
y_train = np.delete(labels,val_vec)

#x_train = np.rollaxis(np.concatenate((x_train1,x_train2),axis=-1),-1,1)[...,np.newaxis]
#x_val = np.rollaxis(np.concatenate((x_val1,x_val2),axis=-1),-1,1)[...,np.newaxis]
x_train = np.concatenate((x_train1,x_train2),axis=-1)
x_val = np.concatenate((x_val1,x_val2),axis=-1)
#%% callbacks
earlyStopping = EarlyStopping(monitor='val_loss',patience=10, verbose=1,mode='auto')

checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
tBoard = TensorBoard(log_dir='./TB_Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
CBs = [earlyStopping,checkpoint,tBoard]
#CBs = []

#%% prepare model for training
redo = 0
if redo==0:
    print("Generating new model")
    CompModel = CompModels.CompModel_v3(x_train)
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    adopt = optimizers.adadelta()
    CompModel.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['binary_accuracy'])
else:
    print('Reloading previous model')
    CompModel = load_model(model_filepath)

Kvis.visualize_graph(CompModel,'CompModelGraph',True)
#%% training
print('Starting training')
numEp = 50
b_s = 4
history = CompModel.fit(x_train,y_train,
                       batch_size=b_s, epochs=numEp,
                       validation_data=(x_val,y_val),
                       verbose=1,callbacks=CBs)

print('Training complete')

#%% get results
predictions = CompModel.predict(x_val,batch_size=20,verbose=1)
scores = CompModel.evaluate(x_val,y_val,batch_size=20)
print('Accuracy: {:0.3f}'.format(scores[1]))
import VisTools
pred_labels = np.where(predictions[...,0]>.5,'positive','negative').tolist()
cor_labels = np.where(y_val>.5,'positive','negative').tolist()
com_labels = [a + '->' + b for a,b in zip(cor_labels,pred_labels)]
side_by_side = np.concatenate((x_val1[...,0],x_val2[...,0]),axis=2)
VisTools.multi_slice_viewer0(side_by_side,com_labels)

if False:
    VisTools.save_labeled_image(side_by_side,com_labels,'ImComp')