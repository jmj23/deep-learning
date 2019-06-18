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
import os
import ants
from tqdm import tqdm

datapath = 'RegNIFTIs/subj{:03d}_{}.nii'
savepath = 'petrecondata_crossval.hdf5'

multiSlice = 3

subj_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
val_num = 3
test_num = 3
train_num = len(subj_vec)-val_num-test_num
np.random.seed(seed=2)

y,x = np.ogrid[-3: 3+1, -3: 3+1]
strel = x**2+y**2 <= 3**2
eps = 1e-12
#%%
def ConvertToMultiSlice(array,MS=3):
    tshp = array.shape
    MSos = np.int((MS-1)/2) #MultiSlice Offset
    MSarray = np.zeros((tshp[0],MS,tshp[1],tshp[2],tshp[3]))
    for ss in range(MSos,tshp[0]-MSos):
        MSarray[ss,0,...] = array[ss-1]
        MSarray[ss,1,...] = array[ss]
        MSarray[ss,2,...] = array[ss+1]
    MSarray[0,0,...] = array[0]
    MSarray[0,1,...] = array[0]
    MSarray[0,2,...] = array[1]
    MSarray[-1,0,...] = array[-2]
    MSarray[-1,1,...] = array[-1]
    MSarray[-1,2,...] = array[-1]
    return MSarray
#%% Inputs
tqdm.write('Loading inputs')

subj = subj_vec[0]
wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
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
#for im in nacims:
#    im[im<0]=0
#    im[im>1500] = 1500
#    im /= 1500
inputarray = np.stack((wims,fims,inims,outims),axis=3)
inputs = inputarray[good_inds]
if multiSlice != 1:
    inputs = ConvertToMultiSlice(inputs,multiSlice)
sliceNums = [inputs.shape[0]]

for subj in tqdm(subj_vec[1:]):
    tqdm.write('Loading subject {}'.format(subj))
    wims = np.rollaxis(ants.image_read(datapath.format(subj,'WATER')).numpy(),2,0)
    fims = np.rollaxis(ants.image_read(datapath.format(subj,'FAT')).numpy(),2,0)
    inims = np.rollaxis(ants.image_read(datapath.format(subj,'InPhase')).numpy(),2,0)
    outims = np.rollaxis(ants.image_read(datapath.format(subj,'OutPhase')).numpy(),2,0)
    good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
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
    if multiSlice != 1:
        new_inputs = ConvertToMultiSlice(new_inputs,multiSlice)
    inputs = np.concatenate((inputs,new_inputs),axis=0)
    sliceNums.append(new_inputs.shape[0])
#%% HU-LAC conversion
def ConvertToLAC(CTims):
    muMap = np.copy(CTims)
    # old conversion
    #muMap[CTims<=0] = (1+CTims[CTims<=0]/1000)*0.096
    #muMap[CTims>0] = (1+CTims[CTims>0]*6.4*10e-4)*0.096
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
    tis_classes[soft_tis] = 2 # soft tissue label
    tis_classes[lung_tis] = 1 # lung tissue label
    tis_classes[bone_tis] = 3 # bone tissue label
    
    n_values = 4
    tis_categorical = np.eye(n_values)[tis_classes]
    return tis_categorical.astype(np.int)

#%%
tqdm.write('Loading targets')

subj = subj_vec[0]
CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
CTims = CTims[good_inds]

# convert HU to LAC
muMap = ConvertToLAC(CTims)

reg_targets = muMap[...,np.newaxis]
class_targets = MakeCTLabels(CTims)

for subj in tqdm(subj_vec[1:]):
    tqdm.write('Loading subject {}'.format(subj))
    CTims = np.rollaxis(ants.image_read(datapath.format(subj,'CTAC')).numpy(),2,0)
    good_inds = np.loadtxt('RegNIFTIs/subj{:03d}_indices.txt'.format(subj)).astype(np.int)
    CTims = CTims[good_inds]
    muMap = ConvertToLAC(CTims)
    new_reg_targets = muMap[...,np.newaxis]
    # make classification array
    new_class_targets = MakeCTLabels(CTims)
    reg_targets = np.concatenate((reg_targets,new_reg_targets),axis=0)
    class_targets = np.concatenate((class_targets,new_class_targets),axis=0)
    
print('---------------------')
print('Slice Nums:')
print(sliceNums)
print('---------------------')
# store validation and testing data to HDF5 file
print('Storing data as HDF5...')
try:
    os.remove(savepath)
except OSError:
    pass
with h5py.File(savepath, 'x') as hf:
    hf.create_dataset("inputs",  data=inputs,dtype='f')
    hf.create_dataset("reg_targets",  data=reg_targets,dtype='f')
    hf.create_dataset("class_targets",  data=class_targets,dtype='f')

print('done')