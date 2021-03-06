# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from os.path import join

import GPUtil
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from natsort import natsorted
from sklearn.model_selection import train_test_split

from DatagenClass import NumpyDataGenerator
from HCC_Models import ResNet50, Inception_model

try:
    if not 'DEVICE_ID' in locals():
            DEVICE_ID = GPUtil.getFirstAvailable()[0]
            print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    print('No GPU available')
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Datapaths
datapath = os.path.expanduser(join(
    '~', 'deep-learning', 'HCC', 'Data'))

# parameters
im_dims = (512, 512)
n_channels = 9
batch_size = 8
epochs = 20
multi_process = False
model_weight_path = 'HCC_classification_model_weights_v1.h5'
val_split = .2


val_params = {'batch_size': batch_size,
              'dim': im_dims,
              'n_channels': n_channels,
              'shuffle': True,
              'rotation_range': 0,
              'width_shift_range': 0.,
              'height_shift_range': 0.,
              'brightness_range': None,
              'shear_range': 0.,
              'zoom_range': 0.,
              'channel_shift_range': 0.,
              'fill_mode': 'constant',
              'cval': 0.,
              'horizontal_flip': False,
              'vertical_flip': False,
              'rescale': None,
              'preprocessing_function': None,
              'interpolation_order': 1}

# Get list of files
pos_dir = join(datapath, 'Positive')
neg_dir = join(datapath, 'Negative')
pos_files = natsorted(glob(join(pos_dir, '*.npy')))
neg_files = natsorted(glob(join(neg_dir, '*.npy')))
# Get subject list
pos_subjs = set([s[-13:-8] for s in pos_files])
neg_subjs = set([s[-13:-8] for s in neg_files])
subj_list = list(pos_subjs.union(neg_subjs))

# split subjects into train/val sets
rng = np.random.RandomState(seed=1)
train_subjs, val_subjs = train_test_split(
    subj_list, test_size=val_split, random_state=rng)
# grab corresponding files for each subject
val_pos_files = [f for f in pos_files if any([s in f for s in val_subjs])]
val_pos_labels = [1.] * len(val_pos_files)
val_neg_files = [f for f in neg_files if any([s in f for s in val_subjs])]
val_neg_labels = [0.] * len(val_neg_files)
val_files = val_pos_files + val_neg_files
val_labels = val_pos_labels + val_neg_labels

val_dict = dict([(f, val_labels[i]) for i, f in enumerate(val_files)])

# Setup datagen
val_gen = NumpyDataGenerator(val_files,
                             val_dict,
                             **val_params)

# Setup model
# HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))

# Load weights
HCCmodel.load_weights(model_weight_path)

# Get predictions
valX,valY = val_gen.__getitem__(1)

preds = HCCmodel.predict_on_batch(valX)

from matplotlib import pyplot as plt

ind = 0
plt.imshow(valX[ind,...,4],cmap='gray')
plt.title('Predicted: {} Actual: {}'.format(preds[ind],valY[ind]))



