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
import keras.backend as K
K.set_image_data_format('channels_last')
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

from HelperFunctions import LoadValData
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
incl_channels = ['T1p','T1a','T1v']
# incl_channels = ['Inp','Out','T2f','T1p','T1a','T1v','T1d','Dw1','Dw2']
n_channels = len(incl_channels)
batch_size = 8
# best_weights_file = 'HCC_best_model_weights.h5'
best_weights_file = 'HCC_best_model_weights_blockmodel.h5'

 # Get datagens
pos_dir = join(datapath, 'Positive')
neg_dir = join(datapath, 'Negative')

x_val, y_val = LoadValData(pos_dir,neg_dir,im_dims,incl_channels)

# Setup model
# HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
# HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))
HCCmodel = BlockModel_Classifier(
        im_dims+(n_channels,), filt_num=8, numBlocks=6)

# Load best weights
HCCmodel.load_weights(best_weights_file)

print('Calculating classification confusion matrix...')
preds = HCCmodel.predict(x_val,batch_size=batch_size, verbose=1)
y_pred = np.rint(preds)
totalNum = len(y_val)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()


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
fpr, tpr, thresholds = roc_curve(y_val, preds, pos_label=1)
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
plt.title('Receiver operating characteristic for HCC')
plt.legend(loc="lower right")
plt.show()


# Get and display predictions
for ind in range(20):
    b_ind = np.random.randint(0,x_val.shape[0])
    cur_im = x_val[b_ind,...]
    cur_pred = preds[b_ind]
    cur_true = y_val[b_ind]
    disp_im = np.concatenate([cur_im[..., c]
                              for c in range(cur_im.shape[-1])], axis=1)
    plt.imshow(disp_im, cmap='gray')
    plt.title('Predicted: {} Actual: {}'.format(cur_pred, cur_true))
    plt.show()
