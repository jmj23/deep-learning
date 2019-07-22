# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=trailing-whitespace
import keras.backend as K
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.initializers import glorot_uniform
from keras.layers import (ELU, Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D,
                          ZeroPadding2D, concatenate)
from keras.models import Model
from keras.regularizers import l2

# from resnet_utils import *

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base +
               '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(
        axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(512, 512, 3), classes=1):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(
        X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(
        X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(
        X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(
        X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def Inception_model(input_shape=(299, 299, 3)):
    incep_model = InceptionV3(
        include_top=False, weights=None, input_shape=input_shape, pooling=None)
    input_layer = incep_model.input
    incep_output = incep_model.output
    x = Conv2D(16, (3, 3), activation='relu')(incep_output)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_layer, outputs=x)


def BlockModel_Classifier(input_shape, filt_num=16, numBlocks=3):
    """Creates a Block model for pretraining on classification task
    Args:
        input shape: a list or tuple of [rows,cols,channels] of input images
        filt_num: the number of filters in the first and last layers
        This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation   
        regression: Whether to have a continuous output with linear activation
    Returns:
        An unintialized Keras model

    Example useage: SegModel = BlockModel2D([256,256,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
    the rows/cols must be divisible by 2^numBlocks for skip connections
    to match up properly
    """

    # check for input shape compatibility
    rows, cols = input_shape[0:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize-np.sum(2**np.arange(1, numBlocks+1)))/2**numBlocks
    assert minsize > 2, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # set l2 regularization parameter
    l2reg = 2e-4

    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # contracting blocks
    x = lay_input
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*(2**(rr-1)), (1, 1), padding='same',
                    name='Conv1_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                    name='Conv3_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                     name='Conv51_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                     name='Conv52_{}'.format(rr), kernel_regularizer=l2(l2reg))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*(2**(rr-1)), (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name='PrePad_{}'.format(rr))(x)
        x = Conv2D(filt_num*(2**(rr-1)), (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr), kernel_regularizer=l2(l2reg))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_skip_{}'.format(rr))(x)

    # average pooling
    x = GlobalAveragePooling2D()(x)
    # classifier
    lay_out = Dense(1, activation='sigmoid', name='output_layer', kernel_regularizer=l2(1e-3))(x)

    return Model(lay_input, lay_out)
