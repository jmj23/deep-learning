# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:04:03 2017

@author: jmj136
"""

import numpy as np
import matplotlib.pyplot as plt
import os
#%%
def showmask(im,mask):
    "Displays the image with the mask overlaid"
    
    msksiz = np.r_[im.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*mask.astype(float)
    
    plt.figure()
    plt.imshow(im,cmap=plt.cm.gray)
    plt.imshow(msk)
    plt.axes = 'off'
    plt.show()
    
#%%
def mask_viewer(imvol,maskvol):
    msksiz = np.r_[maskvol.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*maskvol.astype(float)
    
    fig = plt.figure(figsize=(5,5))
    fig.index = imvol.shape[2] // 2
    imobj = plt.imshow(imvol[:,:,fig.index],cmap='gray',aspect='equal')
    mskobj = plt.imshow(msk[:,:,fig.index,:])
    plt.tight_layout()
    ax = fig.axes[0]
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,fig.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    fig.imvol = imvol
    fig.maskvol = msk
    fig.imobj = imobj
    fig.mskobj = mskobj
    fig.txtobj = txtobj
    fig.canvas.mpl_connect('scroll_event',on_scroll_m)
    
def on_scroll_m(event):
    fig = event.canvas.figure
    if event.button == 'up':
        next_slice_m(fig)
    elif event.button == 'down':
        previous_slice_m(fig)
    fig.txtobj.set_text(fig.index+1)
    fig.canvas.draw()
    
def previous_slice_m(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
#    fig.index = (fig.index - 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index-1,imvol.shape[2]-1]),0])
    fig.imobj.set_data(imvol[:,:,fig.index])
    fig.mskobj.set_data(maskvol[:,:,fig.index,:])
    fig.canvas.draw()

def next_slice_m(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
#    fig.index = (fig.index + 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index+1,imvol.shape[2]-1]),0])
    fig.imobj.set_data(imvol[:,:,fig.index])
    fig.mskobj.set_data(maskvol[:,:,fig.index,:])
    fig.canvas.draw()

#%%
def multi_slice_viewer0(volume,labels):
    fig, ax = plt.subplots()
    if len(volume.shape) != 3:
        print('Volume must be 3D array')
        return
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index,...],cmap='gray',vmin=0, vmax=1)
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,ax.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    ax.txtobj = txtobj
    if 'labels' in locals():
        if len(labels)==volume.shape[0]:
            ax.labels = labels
            lblobj = plt.text(0.9,.98,ax.labels[ax.index],ha='right',va='top',
                               color='yellow',transform=ax.transAxes)
            ax.lblobj = lblobj
            ax.has_labels = True
        else:
            ax.has_labels = False
    else:
        ax.has_labels=False
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.canvas.mpl_connect('scroll_event',on_scroll)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'left' or event.key == 'down':
        previous_slice(ax)
    elif event.key == 'right' or event.key == 'up':
        next_slice(ax)
    fig.canvas.draw()
    
def on_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        next_slice(ax)
    elif event.button == 'down':
        previous_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index - 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)
    if ax.has_labels:
        ax.lblobj.set_text(ax.labels[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index + 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)
    if ax.has_labels:
        ax.lblobj.set_text(ax.labels[ax.index])
    
#%%
def mask_viewer0(imvol,maskvol,name='Mask Display'):
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
    fig.canvas.mpl_connect('scroll_event',on_scroll_m0)
    
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
#    fig.index = (fig.index - 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index-1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

def next_slice_m0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
#    fig.index = (fig.index + 1) % imvol.shape[2]  # wrap around using %
    fig.index = np.max([np.min([fig.index+1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()
#%%
def registration_viewer(fixed,moving,alpha=.5,name='Registration Results'):
    msksiz = np.r_[moving.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = .1
    msk[...,2] = .1
    msk[...,3] = alpha*moving
    
    fig = plt.figure(figsize=(5,5))
    fig.index = 0
    imobj = plt.imshow(fixed[fig.index,...],cmap='gray',aspect='equal')
    mskobj = plt.imshow(msk[fig.index,...])
    plt.tight_layout()
    plt.suptitle(name)
    ax = fig.axes[0]
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,fig.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    fig.imvol = fixed
    fig.maskvol = msk
    fig.imobj = imobj
    fig.mskobj = mskobj
    fig.txtobj = txtobj
    fig.canvas.mpl_connect('scroll_event',on_scroll_r0)
    
def on_scroll_r0(event):
    fig = event.canvas.figure
    if event.button == 'up':
        next_slice_r0(fig)
    elif event.button == 'down':
        previous_slice_r0(fig)
    fig.txtobj.set_text(fig.index+1)
    fig.canvas.draw()
    
def previous_slice_r0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index-1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()

def next_slice_r0(fig):
    imvol = fig.imvol
    maskvol = fig.maskvol
    fig.index = np.max([np.min([fig.index+1,imvol.shape[0]-1]),0])
    fig.imobj.set_data(imvol[fig.index,:,:])
    fig.mskobj.set_data(maskvol[fig.index,:,:,:])
    fig.canvas.draw()    
#%%
def save_masked_image(imvol,maskvol,name='image'):
    msksiz = np.r_[maskvol.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    msk[...,0] = 1
    msk[...,1] = 1
    msk[...,3] = .3*maskvol.astype(float)
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for index in range(imvol.shape[0]):
        fig = plt.figure(0,figsize=(5,5))
        plt.imshow(imvol[index,:,:],cmap='gray',aspect='equal')
        plt.imshow(msk[index,...])
        plt.tight_layout()
        ax = fig.axes[0]
        ax.set_axis_off()
        plt.savefig("{}/{}_{}.png".format(directory,name,index))

#%%
def save_labeled_image(imvol,labels,name='image'):
    directory = 'plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fig = plt.figure(0,figsize=(5,5))
    imobj = plt.imshow(imvol[0,...],cmap='gray',aspect='equal')
    plt.tight_layout()
    ax = fig.axes[0]
    ax.set_axis_off()
    lblobj = plt.text(0.9,.99,labels[0],ha='right',va='top',
                       color='yellow',transform=ax.transAxes)
    
    for index in range(imvol.shape[0]):
        imobj.set_data(imvol[index,...])
        lblobj.set_text(labels[index])
        
        plt.savefig("{}/{}_{}.png".format(directory,name,index))
    