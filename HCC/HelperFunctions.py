from glob import glob
from os.path import join

import cv2
import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from natsort import natsorted
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from DatagenClass import NumpyDataGenerator, PngDataGenerator


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def GetDatagens(pos_dir, neg_dir, batch_size, im_dims, incl_channels):
    val_split = .2
    # Training and validation datagen parameters
    train_params = {'batch_size': batch_size,
                    'dim': im_dims,
                    'n_channels': len(incl_channels),
                    'incl_channels': incl_channels,
                    'shuffle': True,
                    'rotation_range': 15,
                    'width_shift_range': 0.2,
                    'height_shift_range': 0.2,
                    'brightness_range': None,
                    'shear_range': 0.1,
                    'zoom_range': 0.2,
                    'channel_shift_range': 0.,
                    'fill_mode': 'constant',
                    'cval': 0.,
                    'horizontal_flip': True,
                    'vertical_flip': False,
                    'rescale': None,
                    'preprocessing_function': None,
                    'interpolation_order': 1}

    val_params = {'batch_size': batch_size,
                  'dim': im_dims,
                  'n_channels': len(incl_channels),
                  'incl_channels': incl_channels,
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
    pos_files = natsorted(glob(join(pos_dir, '*.png')))
    neg_files = natsorted(glob(join(neg_dir, '*.png')))
    # get list of unique slics
    pos_slices = natsorted(list(set([f[:-7] for f in pos_files])))
    neg_slices = natsorted(list(set([f[:-7] for f in neg_files])))

    # Get subject list
    pos_subjs = natsorted(list(set([s[-9:-5] for s in pos_slices])))
    neg_subjs = natsorted(list(set([s[-9:-5] for s in neg_slices])))

    # split subjects into train/val sets
    rng = np.random.RandomState(seed=1)
    train_pos_subjs, val_pos_subjs = train_test_split(
        pos_subjs, test_size=val_split, random_state=rng)
    train_neg_subjs, val_neg_subjs = train_test_split(
        neg_subjs, test_size=val_split, random_state=rng)
    # grab corresponding files for each subject
    train_pos_files = [f for f in pos_slices if any(
        [s in f for s in train_pos_subjs])]
    train_pos_labels = [1.] * len(train_pos_files)
    train_neg_files = [f for f in neg_slices if any(
        [s in f for s in train_neg_subjs])]
    train_neg_labels = [0.] * len(train_neg_files)
    train_files = train_pos_files + train_neg_files
    train_labels = train_pos_labels + train_neg_labels

    val_pos_files = [f for f in pos_slices if any(
        [s in f for s in val_pos_subjs])]
    val_pos_labels = [1.] * len(val_pos_files)
    val_neg_files = [f for f in neg_slices if any(
        [s in f for s in val_neg_subjs])]
    val_neg_labels = [0.] * len(val_neg_files)
    val_files = val_pos_files + val_neg_files
    val_labels = val_pos_labels + val_neg_labels

    # calculate class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    print('Class weights:')
    print(class_weight_dict)

    # generate label dicts
    train_dict = dict([(f, train_labels[i])
                       for i, f in enumerate(train_files)])
    val_dict = dict([(f, val_labels[i]) for i, f in enumerate(val_files)])

    # Setup datagens
    train_gen = PngDataGenerator(train_files,
                                 train_dict,
                                 **train_params)
    val_gen = PngDataGenerator(val_files,
                               val_dict,
                               **val_params)
    return train_gen, val_gen, class_weight_dict

def LoadSlice(file_root,dims,incl_channels):
    x = np.empty(dims + (len(incl_channels),))
    for s, seq in enumerate(incl_channels):
        # make filename
        cur_file = file_root + seq + '.png'
        # load and resize image
        im = np.array(Image.open(cur_file)).astype(np.float)
        im /= 255.
        if im.shape != dims:
            im = cv2.resize(im, dims)
        x[...,s] = im
    return x

def LoadValData(pos_dir,neg_dir,dims,incl_channels):
    val_split = .2
    # Get list of files
    pos_files = natsorted(glob(join(pos_dir, '*.png')))
    neg_files = natsorted(glob(join(neg_dir, '*.png')))
    # get list of unique slics
    pos_slices = natsorted(list(set([f[:-7] for f in pos_files])))
    neg_slices = natsorted(list(set([f[:-7] for f in neg_files])))

    # Get subject list
    pos_subjs = natsorted(list(set([s[-9:-5] for s in pos_slices])))
    neg_subjs = natsorted(list(set([s[-9:-5] for s in neg_slices])))

    # split subjects into train/val sets
    rng = np.random.RandomState(seed=1)
    _, val_pos_subjs = train_test_split(
        pos_subjs, test_size=val_split, random_state=rng)
    _, val_neg_subjs = train_test_split(
        neg_subjs, test_size=val_split, random_state=rng)

    pos_roots = [f for f in pos_slices if any(
        [s in f for s in val_pos_subjs])]
    neg_roots = [f for f in neg_slices if any(
        [s in f for s in val_neg_subjs])]

    pos_x = np.stack([LoadSlice(f,dims,incl_channels) for f in pos_roots])
    neg_x = np.stack([LoadSlice(f,dims,incl_channels) for f in neg_roots])

    labels = [1.]*pos_x.shape[0] + [0.]*neg_x.shape[0]
    y_val = np.array(labels)
    x_val = np.concatenate((pos_x,neg_x),axis=0)
    return x_val,y_val
