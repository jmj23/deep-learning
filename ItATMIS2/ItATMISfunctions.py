#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:52:53 2018

@author: jmj136
"""

#%% Block Segmentation Model
import numpy as np
from keras.layers import Input,Cropping2D,Conv2D,concatenate
from keras.layers import BatchNormalization,ELU,UpSampling2D
from keras.layers import Conv2DTranspose,ZeroPadding2D
from keras.models import Model

def BlockModel(in_shape,filt_num=16,numBlocks=4,num_out_channels=2):
    input_shape = in_shape[1:]
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    #calculate appropriate cropping
    mod = np.mod(input_shape[0:2],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[0:2]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        numBlocks=3
    
    crop = Cropping2D(cropping=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_input)

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # rest of contracting blocks
    for rr in range(2,numBlocks+1):
        lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # last expanding block
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)    
    lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # rest of expanding blocks
    expnums = list(range(1,numBlocks))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
        
    # segmenter
    lay_out = Conv2D(num_out_channels,(1,1), activation='sigmoid',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)

#%% Dice Coefficient Loss
import keras.backend as K
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

#%% One Pass of iterative training simulation
from keras.preprocessing.image import ImageDataGenerator
def SimulateItATMIS(model,cur_inputs,cur_targets,CBs,val_frac=0.2):
    # split off validation data
    numIm = cur_inputs.shape[0]
    val_inds = np.random.choice(np.arange(numIm),
                                np.round(val_frac*numIm).astype(np.int),
                                replace=False)
    valX = np.take(cur_inputs,val_inds,axis=0)
    valY = np.take(cur_targets,val_inds, axis=0)
    trainX = np.delete(cur_inputs, val_inds, axis=0)
    trainY = np.delete(cur_targets, val_inds, axis=0)
    
    
    # setup image data generator
    datagen1 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    datagen2 = ImageDataGenerator(
        rotation_range=15,
        shear_range=0.5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
     
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    datagen1.fit(trainX, seed=seed)
    datagen2.fit(trainY, seed=seed)
    batchsize = 16
    datagen = zip( datagen1.flow( trainX, None, batchsize, seed=seed), datagen2.flow( trainY, None, batchsize, seed=seed) )
    
    # calculate number of epochs and batches
    # numEp = np.maximum(40,np.minimum(np.int(10*(self.FNind+1)),100))
    numEp = 30
    steps = np.minimum(np.int(trainX.shape[0]/batchsize*16),200)
    
    model.fit_generator(datagen,
                        steps_per_epoch=steps,
                        epochs=numEp,
                        callbacks=CBs,
                        verbose=0,
                        validation_data=(valX,valY))
    return model
#%% Calculate Confidence Interval
import scipy as sp
import scipy.stats

def Calc_Error(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a,axis=0), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

#%% Plot ItATMIS simulation results
from matplotlib import pyplot as plt
from glob import glob
def PlotResults(anatomy,scatter=False):
    txt_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV*.txt'.format(anatomy)
    result_files = glob(txt_path)
    
    scores = [np.loadtxt(f) for f in result_files]
    iters = [range(1,s.shape[0]+1) for s in scores]
    
    plt.figure()
    for it in range(len(scores)):
        if scatter:
            plt.scatter(iters[it],scores[it])
        else:
            plt.plot(iters[it],scores[it],'-o')
        plt.title('Dice Score over Iterations')
        plt.xlabel('Number of subjects')
        plt.ylabel('Dice')
    plt.ylim([0,1])
#%% Plot ItATMIS results as error bars
def PlotErrorResults(anatomy):
    txt_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV*.txt'.format(anatomy)
    result_files = glob(txt_path)
    
    scores = np.stack([np.loadtxt(f) for f in result_files])
    iters = range(1,scores.shape[1]+1)
    m,err = Calc_Error(scores,confidence=.95)
    
    plt.figure()
    plt.errorbar(iters, m, yerr=err, fmt='o',markersize=3,label='ItATMIS')
    plt.title('Dice Score over Iterations')
    plt.xlabel('Number of subjects')
    plt.ylabel('Dice')
    plt.legend()
    plt.ylim([0,1])
#%% Box and whisker plot
def PlotBoxPlot(anatomy,version=1):
    if version==2:
        vs = 'NonItATMIS'
    else:
        vs = 'ItATMIS'
    txt_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/{}_SimResults_{}_CV*.txt'.format(vs,anatomy)
    result_files = glob(txt_path)
    
    scores = np.stack([np.loadtxt(f) for f in result_files],axis=1)
    if version==2:
        iters = [5,10,15,20]
    else:
        iters = range(1,scores.shape[0]+1)
    score_list = [s for s in scores]
    
    # Create a figure instance
    fig = plt.figure(None, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    ax.boxplot(score_list,
               positions=iters,
               patch_artist=True)
    
    plt.title('Dice Score over Iterations for {} data'.format(anatomy))
    plt.xlabel('Number of subjects')
    plt.ylabel('Dice')
    plt.legend([vs])
    plt.xlim([0,21])
    plt.ylim([0,1])
    
    # Save the figure
    fig.savefig('/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/BoxPlot_{}_{}.png'.format(anatomy,vs), bbox_inches='tight')
    
#%% Plot both ItATMIS and NonItATMIS results
def PlotComparison(anatomy):
    # Get files
    itatmis_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV*.txt'.format(anatomy)
    non_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/NonItATMIS_SimResults_{}_CV*.txt'.format(anatomy)
    it_result_files = glob(itatmis_path)
    non_result_files = glob(non_path)
    
    # Create a figure instance
    fig = plt.figure(None, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    
   
    # get itatmis scores
    it_scores = np.stack([np.loadtxt(f) for f in it_result_files])
    it_x = np.arange(1,it_scores.shape[1]+1)
    # tile x array for displaying as one scatter plot
    it_x = np.tile(it_x,(it_scores.shape[0],1))
    
    # get non-itatmis scores
    non_scores = np.stack([np.loadtxt(f) for f in non_result_files])
    non_x = np.array([5,10,15,20])
    # tile x array for displaying as one scatter plot
    non_x = np.tile(non_x,(non_scores.shape[0],1))
    
    ax.scatter(it_x.flatten(),it_scores.flatten(),c='r',label='ItATMIS')
    ax.scatter(non_x.flatten(),non_scores.flatten(),c='b',label='NonItATMIS')
        
    plt.title('Dice Score over Iterations')
    plt.xlabel('Number of subjects')
    plt.ylabel('Dice')
    plt.legend()
    plt.ylim([0,1])
    plt.xlim([0,21])
    
    fig.savefig('/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/Comparison_{}.png'.format(anatomy), bbox_inches='tight')

#%% LCTSC data getter
import os
from skimage.draw import polygon
import pydicom
from skimage.transform import resize

def GetLCTSCdata(directory):
    cur_dir = glob(os.path.join(directory, "*", ""))[0]
    dcm_dir = glob(os.path.join(cur_dir, "0*", ""))[0]
    lbl_dir = glob(os.path.join(cur_dir, "1*", ""))[0]
    dicom_files = glob(os.path.join(dcm_dir, "*.dcm"))
    lbl_file = glob(os.path.join(lbl_dir,"*.dcm"))[0]
    dicms = [pydicom.read_file(fn) for fn in dicom_files]
    dicms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    ims = np.stack([dcm.pixel_array.astype(np.float) for dcm in dicms])
    # normalize
    for im in ims:
        im -= np.mean(im)
        im /= np.std(im)
    # resize
    # pre-allocate array
    inputs = np.zeros((ims.shape[0],256,256))
    # iterate over all the input images and resize
    for i,im in enumerate(ims):
        inputs[i] = resize(ims,(256,256))
        
    # get labels
    label = pydicom.read_file(lbl_file)
    contour_names = [s.ROIName for s in label.StructureSetROISequence]
    # Get the right and left lung indices
    r_ind = contour_names.index('Lung_R')
    l_ind = contour_names.index('Lung_L')
    # Extract the corresponding contours and combine
    contour_right = [s.ContourData for s in label.ROIContourSequence[r_ind].ContourSequence]
    contour_left = [s.ContourData for s in label.ROIContourSequence[l_ind].ContourSequence]
    contours = contour_left + contour_right
    # Z positions
    z = [d.ImagePositionPatient[2] for d in dicms]
    # Rows and columns
    pos_r = dicms[0].ImagePositionPatient[1]
    spacing_r = dicms[0].PixelSpacing[1]
    pos_c = dicms[0].ImagePositionPatient[0]
    spacing_c = dicms[0].PixelSpacing[0]
    # Preallocate
    mask = np.zeros_like(ims)
    # loop over the different slices that each contour is on
    for c in contours:
        nodes = np.array(c).reshape((-1, 3))
        assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        zNew = [round(elem,1) for elem in z]
        try:
            z_index = z.index(nodes[0,2])
        except ValueError:
            z_index = zNew.index(nodes[0,2])
        r = (nodes[:, 1] - pos_r) / spacing_r
        c = (nodes[:, 0] - pos_c) / spacing_c
        rr, cc = polygon(r, c)
        mask[z_index,rr, cc] = 1
        
    # resize
    # pre-allocate array
    targets = np.zeros((mask.shape[0],256,256))
    # iterate over all the target images and resize
    for i,im in enumerate(mask):
        targets[i] = resize(mask,(256,256))   
        
    return inputs,targets

#%% Cardiac MRI Data getter
from scipy import io
def GetCardiacData(image_file,contour_file):
    mat = io.loadmat(image_file)
    image_data = mat['sol_yxzt']
    images = np.rollaxis(np.rollaxis(image_data,3,0),3,1).astype(np.float)
    mask = np.zeros_like(images)
    
    mat = io.loadmat(contour_file)
    contour_data = mat['manual_seg_32points']
    
    ts,zs,_,_ = images.shape
    for t in range(ts):
        for z in range(zs):
            im = images[t,z,...]
            contour = contour_data[z,t]
            if contour[0,0] != -99999:
                numP = int((contour.shape[0]-1)/2)
                endo = contour[:numP,:]
                epi = contour[numP+1:,:]
                
                rr, cc = polygon(epi[:,0], epi[:,1])
                mask[t,z,cc, rr] = 1
                rr, cc = polygon(endo[:,0], endo[:,1])
                mask[t,z,cc, rr] = 0
    
    for imvol in images:
        for im in imvol:
            im -= np.min(im)
            im /= np.max(im)
            
    inputs = np.reshape(images,(-1,256,256))[...,np.newaxis]
    targets = np.reshape(mask,(-1,256,256))[...,np.newaxis]
    return inputs,targets
    
