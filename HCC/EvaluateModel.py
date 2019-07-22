# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-member
import csv
import os
import sys
from glob import glob
from os.path import join
from time import time

import GPUtil
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt
from natsort import natsorted
from natsort import index_natsorted, order_by_index
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from DatagenClass import NumpyDataGenerator
from HCC_Models import Inception_model, ResNet50, BlockModel_Classifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HDF5_USE_FILE_LOCKING'] = 'false'


try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU', DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except Exception as e:
    print('No GPU available')
    print('Using CPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Datapaths
# datapath = os.path.expanduser(join(
    # '~', 'deep-learning', 'HCC', 'Data'))
datapath = join('D:\\', 'jmj136', 'HCCdata')

# parameters
im_dims = (256, 256)
n_channels = 4
incl_channels = [3, 4, 5, 6]
batch_size = 8
# best_weights_file = 'HCC_best_model_weights.h5'
best_weights_file = 'HCC_best_model_weights_blockmodel.h5'
val_split = .2

val_params = {'batch_size': batch_size,
              'dim': im_dims,
              'n_channels': n_channels,
              'incl_channels': incl_channels,
              'shuffle': False,
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
subj_list = natsorted(list(pos_subjs.union(neg_subjs)))

# split subjects into train/val sets
rng = np.random.RandomState(seed=1)
train_subjs, val_subjs = train_test_split(
    subj_list, test_size=val_split, random_state=rng)

val_pos_files = [f for f in pos_files if any([s in f for s in val_subjs])]
val_pos_labels = [1.] * len(val_pos_files)
val_neg_files = [f for f in neg_files if any([s in f for s in val_subjs])]
val_neg_labels = [0.] * len(val_neg_files)
val_files = val_pos_files + val_neg_files
val_labels = val_pos_labels + val_neg_labels

index = index_natsorted(val_files)
val_files = order_by_index(val_files, index)
val_labels = order_by_index(val_labels, index)

# generate label dicts
val_dict = dict([(f, val_labels[i]) for i, f in enumerate(val_files)])

# Setup datagen
val_gen = NumpyDataGenerator(val_files,
                             val_dict,
                             **val_params)

# Setup model
# HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))
HCCmodel = BlockModel_Classifier(
        im_dims+(n_channels,), filt_num=8, numBlocks=6)

# Load best weights
HCCmodel.load_weights(best_weights_file)

print('Calculating classification confusion matrix...')
val_gen.shuffle = False
preds = HCCmodel.predict_generator(val_gen, verbose=1)
labels = [val_gen.labels[f] for f in val_gen.list_IDs]
y_pred = np.rint(preds)
totalNum = len(y_pred)
y_true = np.rint(labels)[:totalNum]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


with open('val_results_eval.txt', 'w') as f:
    for i,j in zip(val_gen.list_IDs[:totalNum],preds[:,0]):
        f.write("{}-{:.04f}\n".format(i,j))
print('Wrote validation results to file')


print('----------------------')
print('Classification Results')
print('----------------------')
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fn)))
print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
print('-----------------------')

# Make ROC curve
fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Get and display predictions
for ind in range(2):
    b_ind = np.random.randint(0,len(val_gen))
    valX, valY = val_gen.__getitem__(b_ind)
    preds = HCCmodel.predict_on_batch(valX)
    cur_im = valX[0]
    disp_im = np.concatenate([cur_im[..., c]
                              for c in range(cur_im.shape[-1])], axis=1)
    plt.imshow(disp_im, cmap='gray')
    plt.title('Predicted: {} Actual: {}'.format(preds[0], valY[0]))
    plt.show()
