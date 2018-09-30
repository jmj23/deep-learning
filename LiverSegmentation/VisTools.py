# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:04:03 2017

@author: jmj136
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
def DisplayDifferenceMask(im,mask1,mask2,name='Difference Mask',savepath=None):
    # adjust masks to binary
    mask1 = (mask1>.5)
    mask2 = (mask2>.5)
    
    # create mask array
    msksiz = np.r_[mask1.shape,4]
    msk = np.zeros(msksiz,dtype=float)
    # calculate mask overlaps
    mask_union = mask1 | mask2
    mask_intersect = mask1 & mask2
    mask1_only = mask1 & ~mask2
    mask2_only = mask2 & ~mask1
    
    # create different colors for overlaps
    msk[mask_intersect,0] = .2
    msk[mask_intersect,1] = .8
    msk[mask_intersect,2] = .2
    
    msk[mask1_only,0] = 1
    msk[mask1_only,1] = 1
    
    msk[mask2_only,1] = .7
    msk[mask2_only,2] = 1
    
    msk[...,3] = .3*mask_union.astype(float)
    
    fig = plt.figure(figsize=(8,8))
    fig.index = 0
    plt.imshow(im,cmap='gray',aspect='equal',vmin=0, vmax=np.max(im))
    plt.imshow(msk)
    plt.tight_layout()
    plt.suptitle(name)
    ax = fig.axes[0]
    ax.set_axis_off()
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
    
#%%
def multi_slice_viewer0(volume,title='',labels=[], vrange=[0,1]):
    # _Inputs_
    # Volume: image volume ndarray in format [slice,rows,columns]
    # title: string to display above images
    # labels: list of labels to display in upper right corner of every slice
    # must have one label per slice
    # vrange: window range in list format: [minimum, maximum]
    
    fig, ax = plt.subplots()
    if len(volume.shape) != 3:
        print('Volume must be 3D array')
        return
    ax.volume = volume
#    ax.index = volume.shape[0] // 2
    ax.index = 0
    ax.imshow(volume[ax.index,...],cmap='gray',vmin=vrange[0], vmax=vrange[1])
    ax.set_title(title)
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
def ROIviewer(volume,coord1,coord2):
    # Volume: image volume ndarray in format [slices,rows,columns]
    
    fig, ax = plt.subplots()
    if len(volume.shape) != 3:
        print('Volume must be 3D array')
        return
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index,...],cmap='gray',vmin=0, vmax=1)
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,ax.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    ax.txtobj = txtobj
    # create ROI
    x = coord1[0]
    y = coord1[1]
    w = coord2[0]-x
    h = coord2[1]-y
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    fig.canvas.mpl_connect('scroll_event',ROIon_scroll)

def ROIon_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        ROInext_slice(ax)
    elif event.button == 'down':
        ROIprevious_slice(ax)
    fig.canvas.draw()

def ROIprevious_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index - 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)

def ROInext_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index + 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)
    
#%%
def MultiROIviewer(volume,coord_array):
    # Volume: image volume ndarray in format [slices,rows,columns]
    # coord_array: [slices,coords]
    # coords: [...,x1,y1,x2,y2,mb,...]
    
    fig, ax = plt.subplots()
    if len(volume.shape) != 3:
        print('Volume must be 3D array')
        return
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index,...],cmap='gray',vmin=0, vmax=1)
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,ax.index+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    ax.txtobj = txtobj
    # get ROI coordinates
    spots = [0,5,10,15]
    roi_data = [np.max(coord_array[:,s:s+5],axis=0) for s in spots]
    # get ROI slices
    roi_inds = [np.where(coord_array[:,s]>0) for s in spots]
    # convert to (x,y,w,h,b/m)
    roi_xywh = [(c[0],c[1],c[2]-c[0],c[3]-c[1],c[4]) for c in roi_data]
    # create rectangle patches
    rects = [patches.Rectangle((c[0],c[1]),c[2],c[3],
                               linewidth=1,
                               edgecolor='r' if c[4]==2 else 'g',
                               facecolor='none') for c in roi_xywh]
    # Add the patch to the Axes
    for rect in rects:
        ax.add_patch(rect)
    ax.rects = rects
    ax.roi_inds = roi_inds
    for ii in range(4):
        if np.any(ax.index==roi_inds[ii][0]):
            rects[ii].set_visible(True)
        else:
            rects[ii].set_visible(False)
            
    fig.canvas.mpl_connect('scroll_event',MultiROIon_scroll)

def MultiROIon_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        MultiROInext_slice(ax)
    elif event.button == 'down':
        MultiROIprevious_slice(ax)
    fig.canvas.draw()

def MultiROIprevious_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index - 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)
    for ii in range(4):
        if np.any(ax.index==ax.roi_inds[ii][0]):
            ax.rects[ii].set_visible(True)
        else:
            ax.rects[ii].set_visible(False)

def MultiROInext_slice(ax):
    volume = ax.volume
    ax.index = np.max([np.min([ax.index + 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index,...])
    ax.txtobj.set_text(ax.index+1)
    for ii in range(4):
        if np.any(ax.index==ax.roi_inds[ii][0]):
            ax.rects[ii].set_visible(True)
        else:
            ax.rects[ii].set_visible(False)

#%%
def CollectYOLO(targ,gw,gh,xi,yi):
    # get center x and y
    bx = gw / (1 + np.exp(-targ[0])) + xi*gw
    by = gh / (1 + np.exp(-targ[1])) + yi*gh
    # get width and height
    bw = gw*np.exp(targ[2])
    bh = gh*np.exp(targ[3])
    # convert center to corner x,y
    bx -= bw/2
    by -= bh/2
    # get class
    malig = np.float(targ[6]>targ[5])
    return bx,by,bw,bh,malig
#%%
def ConvertYOLOtoCoords(target,imdim,conf):
    # look for objects
    (xinds,yinds) = np.where(target[...,4]>conf)
    # get grid parameters
    nx = target.shape[0]
    ny = target.shape[1]
    gw = imdim[0]/nx
    gh = imdim[1]/ny
    # loop over all objects detected
    roi_dat = [CollectYOLO(target[xi,yi,:],gw,gh,xi,yi) for xi,yi in zip(xinds,yinds)]
    return roi_dat
    
#%%
def YOLOviewer(im,target,conf=.5):
    # im: image ndarray in format [rows,columns]
    # target: YOLO target array in format [row,columns,7 data channels]
    # conf: confidence cutoff for objects
    assert len(im.shape)==2
    fig, ax = plt.subplots()
    ax.imshow(im,cmap='gray',vmin=np.min(im), vmax=np.max(im))
    ax.set_axis_off()
    # get coordinates from YOLO target
    roi_xywh = ConvertYOLOtoCoords(target,im.shape,conf)
    # create rectangle patches
    rects = [patches.Rectangle((c[0],c[1]),c[2],c[3],
                               linewidth=1,
                               edgecolor='r' if c[4]==1 else 'g',
                               facecolor='none') for c in roi_xywh]
    # Add the patch to the Axes
    for rect in rects:
        ax.add_patch(rect)
#%%
def mask_viewer0(imvol,maskvol,maskvol2=None,name='Mask Display'):
    if maskvol2 is None:
        # display single mask
        msksiz = np.r_[maskvol.shape,4]
        msk = np.zeros(msksiz,dtype=float)
        msk[...,0] = 1
        msk[...,1] = 1
        msk[...,3] = .3*maskvol.astype(float)
    else:
        assert maskvol2.shape==maskvol.shape
        # mask different mask
        # adjust masks to binary
        maskvol1 = (maskvol>.5)
        maskvol2 = (maskvol2>.5)
        
        # create mask array
        msksiz = np.r_[maskvol1.shape,4]
        msk = np.zeros(msksiz,dtype=float)
        # calculate mask overlaps
        mask_union = maskvol1 | maskvol2
        mask_intersect = maskvol1 & maskvol2
        mask1_only = maskvol1 & ~maskvol2
        mask2_only = maskvol2 & ~maskvol1
        # create different colors for overlaps
        # mask 1 only shows up yellow
        # mask 2 only shows up light blue
        # mask overlap shows up green
        msk[mask_intersect,0] = .2
        msk[mask_intersect,1] = .8
        msk[mask_intersect,2] = .2
        msk[mask1_only,0] = 1
        msk[mask1_only,1] = 1        
        msk[mask2_only,1] = .7
        msk[mask2_only,2] = 1
        msk[...,3] = .3*mask_union.astype(float)
    
    imvol -= np.min(imvol)
    imvol /= np.max(imvol)
    
    fig = plt.figure(figsize=(6,6))
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
def slice_viewer4D(volume,title='', vrange=[0,1]):
    # _Inputs_
    # Volume: image volume 4D ndarray in format [time,slices,rows,columns]
    # title: string to display above images
    # labels: list of labels to display in upper right corner of every slice
    # must have one label per slice
    # vrange: window range in list format: [minimum, maximum]
    
    fig, ax = plt.subplots()
    if len(volume.shape) != 4:
        print('Volume must be 4D array')
        return
    ax.volume = volume
#    ax.index = volume.shape[0] // 2
    ax.index = [0,0]
    ax.imshow(volume[ax.index[0],ax.index[0],...],cmap='gray',vmin=vrange[0], vmax=vrange[1])
    ax.set_title(title)
    ax.set_axis_off()
    txtobj = plt.text(0.05, .95,ax.index[1]+1, ha='left', va='top',color='red',
                      transform=ax.transAxes)
    ax.txtobj = txtobj
    txtobj2 = plt.text(0.05, 0.05,ax.index[0], ha='left', va='top',color='red',
                      transform=ax.transAxes)
    ax.txtobj2 = txtobj2
    
    fig.canvas.mpl_connect('key_press_event', process_key4D)
    fig.canvas.mpl_connect('scroll_event',on_scroll4D)

def process_key4D(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'down':
        previous_slice4D(ax)
    elif event.key == 'up':
        next_slice4D(ax)
    elif event.key == 'left':
        prev_time4D(ax)
    elif event.key == 'right':
        next_time4D(ax)
    fig.canvas.draw()
    
def on_scroll4D(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        next_slice4D(ax)
    elif event.button == 'down':
        previous_slice4D(ax)
    fig.canvas.draw()

def previous_slice4D(ax):
    volume = ax.volume
    ax.index[1] = np.max([np.min([ax.index[1] - 1,volume.shape[1]-1]),0])
    ax.images[0].set_array(volume[ax.index[0],ax.index[1],...])
    ax.txtobj.set_text(ax.index[1]+1)

def next_slice4D(ax):
    volume = ax.volume
    ax.index[1] = np.max([np.min([ax.index[1] + 1,volume.shape[1]-1]),0])
    ax.images[0].set_array(volume[ax.index[0],ax.index[1],...])
    ax.txtobj.set_text(ax.index[1]+1)
        
def prev_time4D(ax):
    volume = ax.volume
    ax.index[0] = np.max([np.min([ax.index[0] - 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index[0],ax.index[1],...])
    ax.txtobj2.set_text(ax.index[0])

def next_time4D(ax):
    volume = ax.volume
    ax.index[0] = np.max([np.min([ax.index[0] + 1,volume.shape[0]-1]),0])
    ax.images[0].set_array(volume[ax.index[0],ax.index[1],...])
    ax.txtobj2.set_text(ax.index[0])
    
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
