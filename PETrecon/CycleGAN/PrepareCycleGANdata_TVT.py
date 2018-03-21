# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:16:21 2017

@author: JMJ136
"""
import h5py
import numpy as np
import skimage.exposure as skexp
from scipy.ndimage import morphology as scimorph
from skimage import measure as skmeasure
import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
import ants

datapath = '~/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_{}.nii'
savepath = 'CycleGAN_data_TVT.hdf5'

subj_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
val_num = 3
test_num = 3
train_num = len(subj_vec)-val_num-test_num
np.random.seed(seed=2)

y,x = np.ogrid[-3: 3+1, -3: 3+1]
strel = x**2+y**2 <= 3**2
eps = 1e-12

#%% Inputs
print('Loading inputs')

subj = subj_vec[0]
wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
#nacims = np.rollaxis(ants.image_read(datapath.format(subj,'NAC')).numpy(),2,0)
good_inds = np.loadtxt('/home/jmj136/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
for im in wims:
    im[im<0]=0
    im /= np.max(im)
for im in fims:
    im[im<0]=0
    im /= np.max(im)
for im in inims:
    im[im<0]=0
    im /= np.max(im)
for im in outims:
    im[im<0]=0
    im /= (np.max(im)+eps)

inputarray = np.stack((wims,fims,inims,outims),axis=3)
inputs = inputarray[good_inds]

sliceNums = [inputs.shape[0]]

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
    good_inds = np.loadtxt('/home/jmj136/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    for im in wims:
        im[im<0]=0
        im /= np.max(im)
    for im in fims:
        im[im<0]=0
        im /= np.max(im)
    for im in inims:
        im[im<0]=0
        im /= np.max(im)
    for im in outims:
        im[im<0]=0
        im /= (np.max(im)+eps)
        
    inputarray = np.stack((wims,fims,inims,outims),axis=3)
    new_inputs = inputarray[good_inds]
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    sliceNums.append(new_inputs.shape[0])
#%% HU-LAC conversion
def ConvertToLAC(CTims):
    muMap = np.copy(CTims)
    # new conversion
    muMap[CTims<=30] = 9.6e-5*(CTims[CTims<=30]+1024)
    muMap[CTims>30]= 5.64e-5*(CTims[CTims>30]+1024)+4.08e-2
    muMap[muMap<0] = 0
    return muMap*5
#%% CT segmentation function
def MakeCTLabels(CTims):
    # make classification array
    soft_tis = np.zeros(CTims.shape,dtype=np.bool)
    soft_tis[CTims>-900] = 1
    for ii in range(soft_tis.shape[0]):
        soft_tis[ii,...] = scimorph.binary_fill_holes(soft_tis[ii,...])
        
    water_tis = np.copy(soft_tis)
    water_tis[CTims<-75] = 0
    fat_tis = np.copy(soft_tis)
    fat_tis[CTims>=-75] = 0
    
    lung_seg = np.zeros(CTims.shape,dtype=np.bool)
    lung_tis = lung_seg
    lung_seg[CTims<-500] = 1
    lung_seg[CTims<-950] = 0
    
    
    for ii in range(lung_tis.shape[0]):
        testim = lung_seg[ii,...]
        all_labels = skmeasure.label(testim)
        properties = skmeasure.regionprops(all_labels)
        areas = np.array([prop.area for prop in properties])
        cents = np.array([prop.centroid for prop in properties])
        sortinds = np.argsort(-areas)
        if np.abs(cents[sortinds[0],1]-cents[sortinds[1],1])>10:
            if cents.shape[0]>2 and np.abs(cents[sortinds[0],1]-cents[sortinds[2],1])<10:
                sortinds[1] = sortinds[2]
            else:
                sortinds[1] = sortinds[0]
        lungmask = all_labels*0
        lungmask[all_labels==sortinds[0]+1] = 1
        lungmask[all_labels==sortinds[1]+1] = 1
        lungmask = scimorph.binary_closing(lungmask,strel)
        lungmask = scimorph.binary_fill_holes(lungmask)
        lung_tis[ii,...] = lungmask.astype(np.bool)
    
    bone_tis = np.zeros(CTims.shape,dtype=np.bool)
    bone_tis[CTims>100] = 1
    
    tis_classes = np.copy(CTims).astype(np.int)
    tis_classes *= 0
    tis_classes[fat_tis] = 2 # fat tissue label
    tis_classes[water_tis] = 3 # water tissue label
    tis_classes[lung_tis] = 1 # lung tissue label
    tis_classes[bone_tis] = 4 # bone tissue label
    
    n_values = 5
    tis_categorical = np.eye(n_values)[tis_classes]
    return tis_categorical.astype(np.int)

#%%
print('Loading targets')

subj = subj_vec[0]
CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
good_inds = np.loadtxt('/home/jmj136/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
CTims = CTims[good_inds]

# convert HU to LAC
muMap = ConvertToLAC(CTims)

reg_targets = muMap[...,np.newaxis]
class_targets = MakeCTLabels(CTims)

for subj in subj_vec[1:]:
    print('Loading subject',subj)
    CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
    good_inds = np.loadtxt('/home/jmj136/deep-learning/PETrecon/RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    CTims = CTims[good_inds]
    muMap = ConvertToLAC(CTims)
    new_reg_targets = muMap[...,np.newaxis]
    # make classification array
    new_class_targets = MakeCTLabels(CTims)
    reg_targets = np.concatenate((reg_targets,new_reg_targets),axis=0)
    class_targets = np.concatenate((class_targets,new_class_targets),axis=0)
    
#%% Split validation,testing data
# get slice ranges corresponding to subjects
sliceNumArray = np.array(sliceNums)
sliceCumArray = np.concatenate(([0],np.cumsum(sliceNumArray)))
# pick out subjects
numSubjs = sliceNumArray.shape[0]
sep_subjs = np.random.choice(np.arange(numSubjs),val_num+test_num,replace=False)
val_subjs = sep_subjs[:val_num]
test_subjs = sep_subjs[val_num:]
train_subjs = np.delete(np.arange(numSubjs),sep_subjs)
# Get validation data
val_inds = np.concatenate(([np.arange(sliceCumArray[ind],sliceCumArray[ind+1]) for ind in val_subjs]))
val_inputs = np.take(inputs,val_inds,axis=0)
val_reg_targets = np.take(reg_targets,val_inds,axis=0)
val_class_targets = np.take(class_targets,val_inds,axis=0)
# Get testing data
test_inds = np.concatenate(([np.arange(sliceCumArray[ind],sliceCumArray[ind+1]) for ind in test_subjs]))
test_inputs = np.take(inputs,test_inds,axis=0)
test_reg_targets = np.take(reg_targets,test_inds,axis=0)
test_class_targets = np.take(class_targets,test_inds,axis=0)
# Remove from training arrays
remove_inds = np.concatenate((test_inds,val_inds))
inputs = np.delete(inputs,remove_inds,axis=0)
reg_targets = np.delete(reg_targets,remove_inds,axis=0)
class_targets = np.delete(class_targets,remove_inds,axis=0)

# store validation and testing data to HDF5 file
print('Storing validation and testing data as HDF5...')
val_inputs = np.rot90(val_inputs,k=1,axes=(1,2))
test_inputs = np.rot90(test_inputs,k=1,axes=(1,2))
val_reg_targets = np.rot90(val_reg_targets,k=1,axes=(1,2))
test_reg_targets = np.rot90(test_reg_targets,k=1,axes=(1,2))
val_class_targets = np.rot90(val_class_targets,k=1,axes=(1,2))
test_class_targets = np.rot90(test_class_targets,k=1,axes=(1,2))
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset("MR_val",  data=val_inputs,dtype='f')
    hf.create_dataset("MR_test", data=test_inputs,dtype='f')
    hf.create_dataset("CT_val_con",  data=val_reg_targets,dtype='f')
    hf.create_dataset("CT_test_con",  data=test_reg_targets,dtype='f')
    hf.create_dataset("CT_val_dis",  data=val_class_targets,dtype='f')
    hf.create_dataset("CT_test_dis",  data=test_class_targets,dtype='f')

#%% augment training data
print('Augmenting training data...')
# LR flips
fl_inputs = np.flip(inputs,1)
fl_reg_targets = np.flip(reg_targets,1)
fl_class_targets = np.flip(class_targets,1)

# gamma corrections
gammas = .5 + np.random.rand(inputs.shape[0])
gm_inputs = np.copy(inputs)
for ii in range(gm_inputs.shape[0]):
    gm_inputs[ii,...,0] = skexp.adjust_gamma(gm_inputs[ii,...,0],gamma=gammas[ii])
    gm_inputs[ii,...,1] = skexp.adjust_gamma(gm_inputs[ii,...,1],gamma=gammas[ii])
    
gm_reg_targets = np.copy(reg_targets)
gm_class_targets = np.copy(class_targets)

# combine together
aug_inputs = np.concatenate((inputs,fl_inputs,gm_inputs),axis=0)
aug_reg_targets = np.concatenate((reg_targets,fl_reg_targets,gm_reg_targets),axis=0)
aug_class_targets = np.concatenate((class_targets,fl_class_targets,gm_class_targets),axis=0)


#%% finalize training data

# store training data
print('Storing train data as HDF5...')
aug_inputs = np.rot90(aug_inputs,k=1,axes=(1,2))
aug_reg_targets = np.rot90(aug_reg_targets,k=1,axes=(1,2))
aug_class_targets = np.rot90(aug_class_targets,k=1,axes=(1,2))
with h5py.File(savepath, 'a') as hf:
    hf.create_dataset("MR_train",  data=aug_inputs,dtype='f')
    hf.create_dataset("CT_train_con",  data=aug_reg_targets,dtype='f')
    hf.create_dataset("CT_train_dis",  data=aug_class_targets,dtype='f')
    
print('done')
#%%
del val_inputs
del val_reg_targets
del val_class_targets

del test_inputs
del test_reg_targets
del test_class_targets

del aug_inputs
del aug_reg_targets
del aug_class_targets

del inputs
del fl_inputs
del gm_inputs
del reg_targets
del class_targets
del fl_reg_targets
del gm_reg_targets
del fl_class_targets
del gm_class_targets
del new_class_targets
del new_reg_targets
del new_inputs
del inputarray