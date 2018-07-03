#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:32:46 2018

@author: jmj136
"""
import os
import glob
import numpy as np
import pydicom
from skimage.draw import polygon
import matplotlib.pyplot as plt

def GetLCTSCdata(directory):
    cur_dir = glob.glob(os.path.join(directory, "*", ""))[0]
    dcm_dir = glob.glob(os.path.join(cur_dir, "0*", ""))[0]
    lbl_dir = glob.glob(os.path.join(cur_dir, "1*", ""))[0]
    dicom_files = glob.glob(os.path.join(dcm_dir, "*.dcm"))
    lbl_file = glob.glob(os.path.join(lbl_dir,"*.dcm"))[0]
    dicms = [pydicom.read_file(fn) for fn in dicom_files]
    dicms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    ims = np.stack([dcm.pixel_array.astype(np.float) for dcm in dicms])
    # normalize
    for im in ims:
        im /= np.max(im)
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
    return ims,mask

#%%
def display_mask(im,mask):
    msksiz = np.r_[mask.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*mask.astype(float)
    
    im -= np.min(im)
    im /= np.max(im)
    
    fig = plt.figure(figsize=(5,5))
    plt.imshow(im,cmap='gray',aspect='equal',vmin=0, vmax=1)
    plt.imshow(msk)
    plt.tight_layout()
    fig.axes[0].set_axis_off()
    plt.show()

#%%
def mask_viewer(imvol,maskvol,name='Mask Display'):
    msksiz = np.r_[maskvol.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*maskvol.astype(float)
    
    imvol -= np.min(imvol)
    imvol /= np.max(imvol)
    
    fig = plt.figure(figsize=(5,5))
    fig.index = 0
    imobj = plt.imshow(imvol[fig.index,...],cmap='gray',aspect='equal',vmin=0, vmax=1)
    mskobj = plt.imshow(msk[fig.index,...])
    plt.tight_layout()
    plt.suptitle(name)
    ax = fig.axes[0]
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,fig.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    fig.imvol = imvol
    fig.maskvol = msk
    fig.imobj = imobj
    fig.mskobj = mskobj
    fig.txtobj = txtobj
#    fig.canvas.mpl_connect('scroll_event',on_scroll_m0)
    
def on_scroll_m0(event):
    fig = event.canvas.figure
    if event.button == 'up':
        next_slice_m0(fig)
    elif event.button == 'down':
        previous_slice_m0(fig)
    fig.txtobj.set_text(fig.index+1)
    fig.canvas.draw()
    
def previous_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index-1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

def next_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index+1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()