# Load in NiFTi file and flip the axis to match the image
import os
import sys
from glob import glob

import nibabel as nib
import numpy as np
from tqdm import tqdm


def process_nifti(image_file, mask_file):
    # Load the mask and data
    mask = nib.load(mask_file)
    mask_hdr = mask.header
    mask_affine = mask.affine

    # load the image and get affine
    image = nib.load(image_file)
    image_affine = image.affine

    mask_data = mask.get_fdata().astype(np.uint8)
    # flip the mask
    mask_data = np.flip(mask_data, axis=0)

    # Create a new mask using affine from image
    new_mask = nib.Nifti1Image(mask_data, image_affine, mask_hdr)
    # covnert datatype of mask to <i2
    new_mask.set_data_dtype("<i2")

    # Save the new image
    fixed_mask_file_path = os.path.join(
        os.path.dirname(mask_file), "fixed_" + os.path.basename(mask_file)
    )
    nib.save(new_mask, fixed_mask_file_path)


# Get the subjects dirs
subject_dirs = glob("/mnt/nas/lawrence/minnhealth_sample/nii/selected_series/*")
for subject_dir in tqdm(subject_dirs, desc="Processing Subjects"):
    # get the only nii file as the image
    image_file = glob(os.path.join(subject_dir, "*.nii"))[0]
    # get the mask file from the ItATMISmasks dir
    mask_file = glob(os.path.join(subject_dir, "ItATMISmasks", "*.nii"))[0]
    # process the nifti
    process_nifti(image_file, mask_file)
