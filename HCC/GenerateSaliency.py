import os
from glob import glob
from os.path import join

import numpy as np
from keras import activations
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from natsort import natsorted
from sklearn.model_selection import train_test_split
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency

from HCC_Models import Inception_model, ResNet50, BlockModel_Classifier

# Datapaths
# datapath = os.path.expanduser(join(
#     '~', 'deep-learning', 'HCC', 'Data'))
datapath = join('D:\\', 'jmj136', 'HCCdata')

# parameters
im_dims = (256, 256)
batch_size = 8
multi_process = True
n_channels = 3
incl_channels = [3, 4, 5]
# best_weights_file = 'HCC_best_model_weights.h5'
best_weights_file = 'HCC_best_model_weights_blockmodel.h5'
val_split = .2

# Data loading
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


# Setup model
# HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))
HCCmodel = BlockModel_Classifier(
        im_dims+(n_channels,), filt_num=8, numBlocks=5)

# get layer index
layer_idx = utils.find_layer_idx(HCCmodel, 'output_layer')
# swap activation
HCCmodel.layers[layer_idx].activation = activations.linear
HCC = utils.apply_modifications(HCCmodel)

# get seed input
seed_input = np.rollaxis(np.load(val_pos_files[0]), 0, 3)
seed_input -= seed_input.mean()
seed_input /= seed_input.std()

# get standard saliency
grads = visualize_saliency(
    HCCmodel, layer_idx, filter_indices=0, seed_input=seed_input)

# try cam visualization
penultimate_layer = utils.find_layer_idx(HCCmodel, 'res5c_branch2c')
grads_cam = visualize_cam(HCCmodel, layer_idx, filter_indices=0, seed_input=seed_input,
                          penultimate_layer_idx=penultimate_layer, backprop_modifier=None)
