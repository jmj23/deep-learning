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
from natsort import natsorted
from sklearn.model_selection import train_test_split

from DatagenClass import NumpyDataGenerator
from ResNet_model import ResNet50

if not 'DEVICE_ID' in locals():
    DEVICE_ID = GPUtil.getFirstAvailable()[0]
    print('Using GPU', DEVICE_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

# Datapaths
datapath = os.path.expanduser(join(
    '~', 'deep-learning', 'HCC', 'Data'))

# parameters
im_dims = (512, 512)
batch_size = 8
epochs = 10
multi_process = True
model_weight_path = 'HCC_classification_model_weights_v1.h5'
val_split = .2

# allow for command-line epochs argument
if len(sys.argv)>1:
    try:
        epochs = int(sys.argv[1])
    except Exception as e:
        pass

# Training and validation datagen parameters
train_params = {'batch_size': batch_size,
                'dim': im_dims,
                'n_channels': 9,
                'shuffle': True,
                'rotation_range': 10,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'brightness_range': None,
                'shear_range': 0.,
                'zoom_range': 0.1,
                'channel_shift_range': 0.,
                'fill_mode': 'constant',
                'cval': 0.,
                'horizontal_flip': True,
                'vertical_flip': True,
                'rescale': None,
                'preprocessing_function': None,
                'interpolation_order': 1}

val_params = {'batch_size': batch_size,
              'dim': im_dims,
              'n_channels': 9,
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
train_pos_files = [f for f in pos_files if any([s in f for s in train_subjs])]
train_pos_labels = [1.] * len(train_pos_files)
train_neg_files = [f for f in neg_files if any([s in f for s in train_subjs])]
train_neg_labels = [0.] * len(train_neg_files)
train_files = train_pos_files + train_neg_files
train_labels = train_pos_labels + train_neg_labels

val_pos_files = [f for f in pos_files if any([s in f for s in val_subjs])]
val_pos_labels = [1.] * len(val_pos_files)
val_neg_files = [f for f in neg_files if any([s in f for s in val_subjs])]
val_neg_labels = [0.] * len(val_neg_files)
val_files = val_pos_files + val_neg_files
val_labels = val_pos_labels + val_neg_labels


train_dict = dict([(f, train_labels[i]) for i, f in enumerate(train_files)])
val_dict = dict([(f, val_labels[i]) for i, f in enumerate(val_files)])

# Setup datagens
train_gen = NumpyDataGenerator(train_files,
                               train_dict,
                               **train_params)
val_gen = NumpyDataGenerator(val_files,
                             val_dict,
                             **val_params)

# Setup model
HCCmodel = ResNet50(input_shape=(512, 512, 9), classes=1)
# compile
HCCmodel.compile(Adam(), loss=binary_crossentropy, metrics=['accuracy'])

HCCmodel.summary()

# Setup callbacks
cb_check = ModelCheckpoint(model_weight_path, monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=True, mode='auto', period=1)
cb_plateau = ReduceLROnPlateau(
    monitor='val_loss', factor=.5, patience=4, verbose=1)


# Train model
print('Starting training...')
history = HCCmodel.fit_generator(generator=train_gen,
                                 epochs=epochs, use_multiprocessing=multi_process,
                                 workers=8, verbose=1, callbacks=[cb_check, cb_plateau],
                                 validation_data=val_gen)
score = HCCmodel.evaluate_generator(val_gen,verbose=1,use_multiprocessing=multi_process,workers=8)
print('|------------------|')
print('|----Results-------|')
print('|------------------|')
print()
print('Accuracy of {:.02f}%'.format(100*score[1]))
print()
print('Done')

