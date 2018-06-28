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