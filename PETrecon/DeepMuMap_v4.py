# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:06:34 2017

@author: JMJ136
"""
import sys
import os
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
#from keras.metrics import mean_absolute_error as mae_metric
#from keras.losses import mean_absolute_error as mae_loss
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import h5py
import time
from CustomMetrics import weighted_mse
import GPUtil
if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU',DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

numEp = 50
b_s = 4
dual_output = False
filterMult = 8
#%%
# Model Save Path/name
model_filepath = 'MuMapModel_test.hdf5'
# Data path/name
datapath = 'petrecondata_tvt_v4.hdf5'
MSos = 1    # MultiSlice offset

with h5py.File(datapath,'r') as f:
    x_train = np.array(f.get('train_inputs'))
    y_reg_train = np.array(f.get('train_reg_targets'))
    y_class_train = np.array(f.get('train_class_targets'))
    trainSliceInds = np.array(f.get('trainSliceInds'))
    x_val = np.array(f.get('val_inputs'))
    y_reg_val = np.array(f.get('val_reg_targets'))
    y_class_val = np.array(f.get('val_class_targets'))
    valSliceInds = np.array(f.get('valSliceInds'))
    x_test = np.array(f.get('test_inputs'))
    y_reg_test = np.array(f.get('test_reg_targets'))    
    y_class_test = np.array(f.get('test_class_targets'))
    testSliceInds = np.array(f.get('testSliceInds'))
    
#%% Model
from keras.layers import Input, Conv2D, concatenate, Conv3D#, Cropping2D
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import ELU
from keras.models import Model

def BlockModel_reg(samp_input,dual_output,fnum=8):
    lay_input = Input(shape=(samp_input.shape[1:]),name='input_layer')
    
    padamt = 1
    MSconv = Conv3D(16,(3,3,3),padding='valid',name='MSconv')(lay_input)
    bn = BatchNormalization()(MSconv)
    MSact = ELU(name='MSelu')(bn)
    MSconvRS = Reshape((254,254,16))(MSact)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(fnum*(2**(rr-1)), (1, 1),padding='same',name='Conv1_{}'.format(rr))(MSconvRS)
    lay_conv3 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv3_{}'.format(rr))(MSconvRS)
    lay_conv51 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv51_{}'.format(rr))(MSconvRS)
    lay_conv52 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(fnum*(2**(rr-1)),(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(fnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,4):
        lay_conv1 = Conv2D(fnum*(2**(rr-1)), (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(fnum*(2**(rr-1)), (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(fnum*(2**(rr-1)),(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(fnum*(2**(rr-1)),(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=3
    lay_deconv1 = Conv2D(fnum*(2**(dd-1)),(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(fnum*(2**(dd-1)),(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(fnum*(2**(dd-1)), (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(fnum*(2**(dd-1)), (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(fnum*(2**(dd-1)),(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)
#    lay_stride = Conv2DTranspose(16*dd,(4,4),strides=(2,2),name='DeConvStride_{}'.format(dd))(lay_act)
#    lay_act = ELU(name='elu_d{}_2'.format(dd))(lay_stride)
    
    lay_cleanup = Conv2DTranspose(fnum*(2**(dd-1)), (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(fnum*(2**(dd-1)), (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(fnum*(2**(dd-1)),(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(fnum*(2**(dd-1)),(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(fnum*(2**(dd-1)), (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(fnum*(2**(dd-1)), (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(fnum*(2**(dd-1)),(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(fnum*(2**(dd-1)), (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(fnum*(2**(dd-1)), (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
        
    # regressor
    lay_reg = Conv2D(1,(1,1), activation='linear',name='reg_output')(lay_pad)
    if dual_output:
        # classifier
        lay_class = Conv2D(4,(1,1), activation='softmax',name='class_output')(lay_pad)
        returnModel = Model(inputs=[lay_input],outputs=[lay_reg,lay_class])
    else:
        returnModel = Model(lay_input,lay_reg)
        
    return returnModel
    
#%% callbacks
#earlyStopping = EarlyStopping(monitor='val_reg_output_loss',patience=10,verbose=1,mode='auto')
if dual_output:
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_reg_output_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)
else:
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss',verbose=0,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

#CBs = [checkpoint,earlyStopping]
CBs = [checkpoint]

#%% prepare model for training
print("Generating model")

RegModel = BlockModel_reg(x_train,dual_output,filterMult)
adopt = optimizers.adam()
if dual_output:
    RegModel.compile(optimizer=adopt,
                 loss={'reg_output': weighted_mse, 'class_output': "categorical_crossentropy"},
                 loss_weights={'reg_output': 1., 'class_output': .3})
else:
    RegModel.compile(optimizer=adopt,loss= weighted_mse)

#%% training
print('Starting training')

if dual_output:
    history = RegModel.fit(x_train,
                       {'reg_output': y_reg_train,'class_output':y_class_train},
                       batch_size=b_s, epochs=numEp,shuffle=True,
                       validation_data=(x_val,{'reg_output': y_reg_val,'class_output':y_class_val}),
                       verbose=1,
                       callbacks=CBs)
else:
    history = RegModel.fit(x_train,y_reg_train,
                       batch_size=b_s, epochs=numEp,shuffle=True,
                       validation_data=(x_val, y_reg_val),
                       verbose=1,
                       callbacks=CBs)

print('Training complete')

print('Loading best model...')
RegModel = load_model(model_filepath,custom_objects={'weighted_mse':weighted_mse})

score = RegModel.evaluate(x_test,{'reg_output': y_reg_test,'class_output':y_class_test})
print("")
print("Metrics on test data: {}".format(score))

#%%
print('Generating samples')
# regression result
pr_bs = np.minimum(16,x_test.shape[0])
time1 = time.time()
output = RegModel.predict(x_test,batch_size=pr_bs)
time2 = time.time()
print('Infererence time: ',1000*(time2-time1)/x_test.shape[0],' ms per slice')
if dual_output:
    reg_output = output[0]
    class_output = output[1]
else:
    reg_output = output


#val_output = RegModel.predict(x_val,batch_size=pr_bs)
#val_reg_output = val_output[0]
#val_class_output = val_output[1]
#
#from skimage.measure import compare_ssim as ssim
#SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_reg_test[...,0],reg_output[...,0])]
#val_SSIMs = [ssim(im1,im2) for im1, im2 in zip(y_reg_val[...,0],val_reg_output[...,0])]
#
#num_bins = 10
#fig3 = plt.figure()
#n, bins, _ = plt.hist(SSIMs, num_bins, facecolor='blue', edgecolor='black', alpha=0.5)
#plt.show()
#print('Mean SSIM of ', np.mean(SSIMs))
#print('SSIM range of ', np.round(np.min(SSIMs),3), ' - ', np.round(np.max(SSIMs),3))
#
## process classification result
#val_class_inds = np.argmax(val_class_output,axis=3)
#yval_class_inds = np.argmax(y_class_val,axis=3)
#test_class_inds = np.argmax(class_output,axis=3)
#ytest_class_inds = np.argmax(y_class_test,axis=3)
#
#from VisTools import multi_slice_viewer0
#multi_slice_viewer0(np.c_[x_test[:,MSos,...,0],reg_output[...,0],y_reg_test[...,0],ytest_class_inds/3,test_class_inds/3],'Test Data',SSIMs)
#multi_slice_viewer0(np.c_[x_val[:,MSos,...,0],val_reg_output[...,0],y_reg_val[...,0],yval_class_inds/3,val_class_inds/3],'Val Data',val_SSIMs)

mae = np.mean(np.abs(reg_output-y_reg_test))
print('Mean Absolute Error is:',mae)
#%% plotting
print('Plotting metrics')
step = np.minimum(b_s/x_train.shape[0],1)
actEpochs = len(history.history['loss'])
epochs = np.arange(1,actEpochs+1)

fig2 = plt.figure(1,figsize=(12.0, 6.0));
if dual_output:
    plt.semilogy(epochs,history.history['reg_output_loss'],'r-s')
    plt.semilogy(epochs,history.history['val_reg_output_loss'],'m-s')
else:
    plt.semilogy(epochs,history.history['loss'],'r-s')
    plt.semilogy(epochs,history.history['val_loss'],'m-s')
plt.legend(['Regression Training Loss',
            'Regression Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.show()

if dual_output:
    np.savetxt('Loss_Dual.txt',np.c_[history.history['reg_output_loss'],history.history['val_reg_output_loss']])
else:
    np.savetxt('Loss_noDual.txt',np.c_[history.history['loss'],history.history['val_loss']])

#%%
data = np.loadtxt('Loss_noDual.txt')
nodual_train = data[:,0]
nodual_val = data[:,1]
data = np.loadtxt('Loss_Dual.txt')
dual_train = data[:,0]
dual_val = data[:,1]

fig3 = plt.figure(2,figsize=(12.0, 6.0));
plt.semilogy(epochs,dual_train,'r-')
plt.semilogy(epochs,dual_val,'rs')
plt.semilogy(epochs,nodual_train,'b-')
plt.semilogy(epochs,nodual_val,'bs')
plt.legend(['Dual Training Loss',
            'Dual Validation Loss',
            'NonDual Training Loss',
            'NonDual Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.show()