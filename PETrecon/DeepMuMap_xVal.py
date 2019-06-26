# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.chdir(os.path.join(os.path.expanduser('~'),'deep-learning','PETrecon'))
sys.path.insert(1,os.path.join(os.path.expanduser('~'),'deep-learning','Utils'))
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.losses import mean_absolute_error as mae_loss
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from CustomMetrics import weighted_mae
from HelperFunctions import BlockModel_reg
import skimage.exposure as skexp
# Get the first available GPU
import GPUtil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])

dual_output = False
numEp = 50
numFolds = 6

#%%
# Model Save Path/name
if dual_output:
    model_filepath = 'MuMapModel_Xval_fold_{}.hdf5'
else:
    model_filepath = 'MuMapModel_Xval_fold_{}_nodual.hdf5'
# Data path/name
datapath = 'petrecondata_crossval.hdf5'
MSos = 1    # MultiSlice offset

with h5py.File(datapath, 'r') as f:
    x = np.array(f.get('inputs'))
    y_reg = np.array(f.get('reg_targets'))
    y_class = np.array(f.get('class_targets'))

# split data into list of arrays, subject-wise
sliceNumArray = np.array([59, 50, 52, 47, 50, 55, 51, 50, 50, 58, 59, 52, 60, 59, 55, 53, 55, 52])
sliceCumArray = np.concatenate(([0],np.cumsum(sliceNumArray)))
x_list = [x[sliceCumArray[ind]:sliceCumArray[ind+1]] for ind in range(len(sliceNumArray))]
y_reg_list = [y_reg[sliceCumArray[ind]:sliceCumArray[ind+1]] for ind in range(len(sliceNumArray))]
y_class_list = [y_class[sliceCumArray[ind]:sliceCumArray[ind+1]] for ind in range(len(sliceNumArray))]
del x,y_reg,y_class

def GetCurrentSplits(numFolds,cur_fold,x_list,y_reg_list,y_class_list):
    # split into cross validation folds
    lst = range(18)
    subj_groups = np.array_split(lst,numFolds)
    cur_val_group = cur_fold
    cur_test_group = (cur_fold + 1) % numFolds
    cur_train_group = list(range(numFolds))
    cur_train_group.remove(cur_val_group)
    cur_train_group.remove(cur_test_group)
    cur_train_subjs = np.concatenate([g for i,g in enumerate(subj_groups) if i in cur_train_group])
    cur_val_subjs = subj_groups[cur_val_group]
    cur_test_subjs = subj_groups[cur_test_group]
    cur_train_x = np.concatenate([x_list[ind] for ind in cur_train_subjs])
    cur_val_x = np.concatenate([x_list[ind] for ind in cur_val_subjs])
    cur_test_x = np.concatenate([x_list[ind] for ind in cur_test_subjs])
    cur_train_y_reg = np.concatenate([y_reg_list[ind] for ind in cur_train_subjs])
    cur_val_y_reg = np.concatenate([y_reg_list[ind] for ind in cur_val_subjs])
    cur_test_y_reg = np.concatenate([y_reg_list[ind] for ind in cur_test_subjs])
    cur_train_y_class = np.concatenate([y_class_list[ind] for ind in cur_train_subjs])
    cur_val_y_class = np.concatenate([y_class_list[ind] for ind in cur_val_subjs])
    cur_test_y_class = np.concatenate([y_class_list[ind] for ind in cur_test_subjs])
    x_tup = cur_train_x,cur_val_x,cur_test_x
    y_tup = cur_train_y_reg,cur_val_y_reg,cur_test_y_reg,cur_train_y_class,cur_val_y_class,cur_test_y_class
    return x_tup,y_tup

#%% Cross validation training
if dual_output:
    scores = np.zeros((numFolds,5))
else:
    scores = np.zeros((numFolds,3))

for fold in range(numFolds):

    print('Starting fold {}/{}...'.format(fold+1,numFolds))
    # Get current data
    print('Getting current fold data...')
    (trainX,valX,testX),(trainYr,valYr,testYr,trainYc,valYc,testYc) = GetCurrentSplits(numFolds,fold,x_list,y_reg_list,y_class_list)
    # Set callbacks
    cur_filepath = model_filepath.format(fold)
#     earlyStopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1,mode='auto')
    if dual_output:
        checkpoint = ModelCheckpoint(cur_filepath, monitor='val_reg_output_loss',verbose=0,
                                     save_best_only=True, save_weights_only=True,
                                     mode='auto', period=1)
    else:
        checkpoint = ModelCheckpoint(cur_filepath, monitor='val_loss',verbose=0,
                                    save_best_only = True, save_weights_only = True,
                                    mode='auto',period=1)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1,factor=0.5)
    CBs = [checkpoint,reduceLR]

    # augment training data
    print('Augmenting training data...')
    # LR flips
    flX = np.flip(trainX,2)
    flYr = np.flip(trainYr,1)
    if dual_output:
        flYc = np.flip(trainYc,1)

    # gamma corrections
    gammas = .5 + np.random.rand(trainX.shape[0])
    gmX = np.copy(trainX)
    for ii in range(gmX.shape[0]):
        gmX[ii,...,0] = skexp.adjust_gamma(gmX[ii,...,0],gamma=gammas[ii])
        gmX[ii,...,1] = skexp.adjust_gamma(gmX[ii,...,1],gamma=gammas[ii])

    gmYr = np.copy(trainYr)
    if dual_output:
        gmYc = np.copy(trainYc)

    # combine all together
    trainX = np.concatenate((trainX,flX,gmX))
    trainYr = np.concatenate((trainYr,flYr,gmYr))
    if dual_output:
        trainYc = np.concatenate((trainYc,flYc,gmYc))

    # prepare model for training
    print("Generating new model")

    RegModel = BlockModel_reg(trainX,dual_output,8)
    adopt = optimizers.adam()
    if dual_output:
        RegModel.compile(optimizer=adopt,
                     loss={'reg_output': weighted_mae, 'class_output': "categorical_crossentropy"},
                     loss_weights={'reg_output': 1., 'class_output': .3},
                     metrics={'reg_output':mae_loss})
    else:
        RegModel.compile(optimizer=adopt,loss= weighted_mae,metrics=[mae_loss])

    # training
    print('Starting training')
    if dual_output:
        history = RegModel.fit(trainX,
                               {'reg_output': trainYr,'class_output':trainYc},
                               batch_size=16, epochs=numEp,shuffle=True,
                               validation_data=(valX,{'reg_output': valYr,'class_output':valYc}),
                               verbose=1,
                               callbacks=CBs)
    else:
        history = RegModel.fit(trainX,trainYr,
                               batch_size=16, epochs=numEp,shuffle=True,
                               validation_data=(valX,valYr),
                               verbose=1,
                               callbacks=CBs)
    print('Training complete')

    print('Loading best model...')
    try:
        RegModel.load_weights(cur_filepath)
    except Exception as e:
        print('No new model saved')
    if dual_output:
        print('Evaluating testing set...')
        test_score = RegModel.evaluate(testX,{'reg_output': testYr,'class_output': testYc})
        print('')
        print("Metrics on test data for fold {}".format(fold+1))
        print("Weighted L1 loss: {:.04e}, Classification Loss: {:.04e}".format(test_score[1],test_score[2]))
        print('Mean absolute error is: {:.04e}'.format(test_score[3]))
        scores[fold] = [fold] + test_score
    else:
        print('Evaluating testing set...')
        test_score = RegModel.evaluate(testX,testYr)
        print('')
        print("Metrics on test data for fold {}".format(fold+1))
        print("Weighted L1 loss for fold {}: {:.04e}".format(fold,test_score[0]))
        print('Mean absolute error is: {:.04e}'.format(test_score[1]))
        scores[fold] = [fold] + test_score

    # Save score data
    if dual_output:
        np.savetxt('cross_validation_scores.txt', scores, fmt='%.08f')
    else:
        np.savetxt('cross_validation_scores_nodual.txt', scores, fmt='%.08f')
    print('Data saved')

print('Cross validation complete')
#%% Display results of training
from tabulate import tabulate
if dual_output:
    scores = np.loadtxt('cross_validation_scores.txt', dtype=float)
    print(tabulate(scores, headers=['Fold #','Comb Score', 'Reg Score', 'Class Score','MAE'],tablefmt='fancy_grid'))
else:
    scores = np.loadtxt('cross_validation_scores_nodual.txt',dtype=float)
    print(tabulate(scores, headers=['Fold #','Reg Score','MAE'],tablefmt='fancy_grid'))
