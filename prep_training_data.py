# load in nii image and mask data
# prep data for training
# Data must be saved in a single HDF5 file in the following format:
# two datasets: inputs "x" and targets "y"
# inputs are of shape (samples,row,col,channels)
# outputs are of shape (samples,row,col,classes)
import os
from glob import glob

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm

from Utils.VisTools import save_masked_image

# data is stored here, one directory per subject
# each subject has a "ItATMISmasks" subdirectory with the mask nii file
data_dir = "/mnt/nas/lawrence/minnhealth_sample/nii/selected_series/"

# for output
project_dir = "/mnt/nas/jacob/minnhealth"
datapath = os.path.join(project_dir, "SegData.hdf5")

# load in the image and mask data
subject_dirs = sorted(glob(data_dir + "*/"))
num_subjects = len(subject_dirs)
print(f"Found {num_subjects} subjects")

image_arrays = []
mask_arrays = []
for subject_dir in tqdm(subject_dirs, desc="Loading data"):
    mask_dir = os.path.join(subject_dir, "ItATMISmasks/")
    mask_files = sorted(glob(mask_dir + "*.nii"))
    mask_file = mask_files[0]
    mask = nib.load(mask_file)
    canon_nft = nib.as_closest_canonical(mask)
    mask = np.swapaxes(np.rollaxis(canon_nft.get_fdata(), 2, 0), 1, 2)
    # count slices with positive mask
    pos_slices = np.sum(mask, axis=(1, 2)) > 0
    neg_slices = np.logical_not(pos_slices)

    # remove some slices with no mask so there are equal numbers of positive and negative slices
    # num_pos = np.sum(pos_slices)
    # num_neg = np.sum(neg_slices)
    # num_remove = num_neg - num_pos
    # remove_indices = np.where(neg_slices)[0][np.random.choice(num_neg, num_remove, replace=False)]

    # remove all slices with no mask
    remove_indices = np.where(neg_slices)[0]

    mask = np.delete(mask, remove_indices, axis=0)
    mask_arrays.append(mask)
    image_files = sorted(glob(subject_dir + "*.nii"))
    assert len(image_files) == 1
    image_file = image_files[0]
    nft = nib.load(image_file)
    canon_nft = nib.as_closest_canonical(nft)
    image = np.swapaxes(np.rollaxis(canon_nft.get_fdata(), 2, 0), 1, 2).astype(np.float32)
    # apply the same slice removal to the image
    image = np.delete(image, remove_indices, axis=0)
    # apply clipping
    image = np.clip(image, -1000, 2000)
    image = image + 1000
    image = image / 3000
    image_arrays.append(image)

# verify that the image and mask shapes match
save_masked_image(image_arrays[0], mask_arrays[0])

# concatenate the image arrays and add channel dim
inputs = np.concatenate(image_arrays, axis=0)[..., np.newaxis]
# concatenate the mask arrays
targets = np.concatenate(mask_arrays, axis=0)
# convert to one-hot encoding
targets = np.eye(3)[targets.astype(int)]

# store to HDF5 file
print("storing as HDF5")
with h5py.File(datapath, "w") as f:
    f.create_dataset("x", data=inputs)
    f.create_dataset("y", data=targets)

print("done")
