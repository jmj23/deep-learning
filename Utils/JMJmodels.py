#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:02:22 2018

@author: Jacob Johnson, MS
@email: jmjohnson33@wisc.edu
@url: https://github.com/jmj23/deep-learning/

All rights reserved
"""

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Cropping2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    ZeroPadding3D,
    add,
    concatenate,
)
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.models import Model

# conv_initD = RandomNormal(0, 0.02)
conv_initD = "he_normal"


# GAN batch normalization
def batchnorm():
    gamma_init = RandomNormal(1.0, 0.02)
    return BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5, gamma_initializer=gamma_init)


use_bn = False


# %% Parameterized 2D Block Model
def BlockModel2D(input_shape, filt_num=16, numBlocks=3, output_chan=1, regression=False):
    """Creates a Block CED model for segmentation or image regression problems

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
    rows, cols = input_shape[:2]
    assert rows % 2**numBlocks == 0, "Input rows and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input cols and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.max(input_shape[0:2])
    minsize = (startsize - np.sum(2 ** np.arange(1, numBlocks + 1))) / 2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name="input_layer")

    # contracting blocks
    x = lay_input
    skip_list = []
    for rr in range(1, numBlocks + 1):
        x1 = Conv2D(filt_num * rr, (1, 1), padding="same", name="Conv1_{}".format(rr))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_x1_{}".format(rr))(x1)
        x3 = Conv2D(filt_num * rr, (3, 3), padding="same", name="Conv3_{}".format(rr))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_x3_{}".format(rr))(x3)
        x51 = Conv2D(filt_num * rr, (3, 3), padding="same", name="Conv51_{}".format(rr))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_x51_{}".format(rr))(x51)
        x52 = Conv2D(filt_num * rr, (3, 3), padding="same", name="Conv52_{}".format(rr))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_x52_{}".format(rr))(x52)
        x = concatenate([x1, x3, x52], name="merge_{}".format(rr))
        x = Conv2D(filt_num * rr, (1, 1), padding="valid", name="ConvAll_{}".format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_all_{}".format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name="PrePad_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (4, 4),
            padding="valid",
            strides=(2, 2),
            name="DownSample_{}".format(rr),
        )(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_downsample_{}".format(rr))(x)
        x = Conv2D(filt_num * rr, (3, 3), padding="same", name="ConvClean_{}".format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_clean_{}".format(rr))(x)
        print(x.shape[1])
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks + 1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            print(skip_list[dd - 1].shape[1], x.shape[1])
            x = concatenate([skip_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x1 = Conv2D(filt_num * dd, (1, 1), padding="same", name="DeConv1_{}".format(dd))(x)
        x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_Dx1_{}".format(dd))(x1)
        x3 = Conv2D(filt_num * dd, (3, 3), padding="same", name="DeConv3_{}".format(dd))(x)
        x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_Dx3_{}".format(dd))(x3)
        x51 = Conv2D(filt_num * dd, (3, 3), padding="same", name="DeConv51_{}".format(dd))(x)
        x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_Dx51_{}".format(dd))(x51)
        x52 = Conv2D(filt_num * dd, (3, 3), padding="same", name="DeConv52_{}".format(dd))(x51)
        x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_Dx52_{}".format(dd))(x52)
        x = concatenate([x1, x3, x52], name="Dmerge_{}".format(dd))
        x = Conv2D(filt_num * dd, (1, 1), padding="valid", name="DeConvAll_{}".format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_Dall_{}".format(dd))(x)
        x = UpSampling2D(size=(2, 2), name="UpSample_{}".format(dd))(x)
        x = Conv2D(filt_num * dd, (3, 3), padding="same", name="DeConvClean1_{}".format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean1_{}".format(dd))(x)
        x = Conv2D(filt_num * dd, (3, 3), padding="same", name="DeConvClean2_{}".format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean2_{}".format(dd))(x)

    # classifier
    if regression:
        if residual:
            lay_res = Conv2D(1, (1, 1), activation="linear", name="regression_layer")(x)
            in0 = Lambda(lambda x: x[..., 1, 0], name="channel_split")(lay_input)
            in0 = Reshape([rows, cols, 1])(in0)
            lay_out = add([in0, lay_res], name="output_layer")
        else:
            lay_out = Conv2D(output_chan, (1, 1), activation="linear", name="regression_layer")(x)
    elif output_chan == 1:
        lay_out = Conv2D(output_chan, (1, 1), activation="sigmoid", name="output_layer")(x)
    else:
        lay_out = Conv2D(output_chan, (1, 1), activation="softmax", name="output_layer")(x)

    return Model(lay_input, lay_out)


# %% Parameterized 2.5D Block Model with 3 input slices
def BlockModelMS3(
    input_shape,
    filt_num=16,
    numBlocks=3,
    output_chan=1,
    regression=False,
    residual=False,
    use_bn=False,
    conv_init="glorot_uniform",
):
    """Creates a Block CED model for segmentation or image regression problems
    with 3 slice input and 1 slice output

    Args:
        input shape: a list or tuple of [rows,cols,slices,channels] of input images
        filt_num: the number of filters in the first and last layers
            This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
        residual: Whether to create a ResNet where the network learns a residual to add
            to the input. The output of the network is added to the first channel of the
            middle slice of the input to generate the final output.
        use_bn: Whether to use batch normalization. Not recommended for GAN useage
        conv_init: The initializer string to use for convolution kernel initialization.


    Returns:
        An unintialized Keras model

    Example useage: GenModel = BlockModelMS3([256,256,3,1],filt_num=8)

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
        the rows/cols must be divisible by 2^numBlocks for skip connections
        to match up properly

    """

    # check for input shape compatibility
    rows, cols, slices = input_shape[0:3]
    assert slices == 3, "Number of slices must be 3"
    assert rows % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.maximum(rows, cols)
    minsize = (startsize - np.sum(2 ** np.arange(1, numBlocks + 1))) / 2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name="input_layer")

    MSpad = ZeroPadding3D(padding=(1, 1, 0), name="MSprepad")(lay_input)
    MSconv = Conv3D(
        filt_num,
        (3, 3, 3),
        padding="valid",
        name="MSconv1",
        kernel_initializer=conv_init,
    )(MSpad)
    if use_bn:
        bn = batchnorm()(MSconv)
        MSact = ELU(name="MSelu1")(bn)
    else:
        MSact = ELU(name="MSelu1")(MSconv)

    x = Reshape((rows, cols, filt_num))(MSact)
    print(x.shape)

    # contracting blocks
    skip_list = []
    for rr in range(1, numBlocks + 1):
        x1 = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="same",
            name="Conv1_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_x1_{}".format(rr))(x1)
        x3 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv3_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_x3_{}".format(rr))(x3)
        x51 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv51_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_x51_{}".format(rr))(x51)
        x52 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv52_{}".format(rr),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_x52_{}".format(rr))(x52)
        x = concatenate([x1, x3, x52], name="merge_{}".format(rr))
        x = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="valid",
            name="ConvAll_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_all_{}".format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name="PrePad_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (4, 4),
            padding="valid",
            strides=(2, 2),
            name="DownSample_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_downsample_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="ConvClean_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_clean_{}".format(rr))(x)
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks + 1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x1 = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="same",
            name="DeConv1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_Dx1_{}".format(dd))(x1)
        x3 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv3_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_Dx3_{}".format(dd))(x3)
        x51 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv51_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_Dx51_{}".format(dd))(x51)
        x52 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv52_{}".format(dd),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_Dx52_{}".format(dd))(x52)
        x = concatenate([x1, x3, x52], name="Dmerge_{}".format(dd))
        x = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="valid",
            name="DeConvAll_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dall_{}".format(dd))(x)
        x = UpSampling2D(size=(2, 2), name="UpSample_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean1_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean2_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean2_{}".format(dd))(x)

    # classifier
    if regression:
        if residual:
            lay_res = Conv2D(
                1,
                (1, 1),
                activation="linear",
                name="regression_layer",
                kernel_initializer=conv_init,
            )(x)
            in0 = Lambda(lambda x: x[..., 1, 0], name="channel_split")(lay_input)
            in0 = Reshape([rows, cols, 1])(in0)
            lay_out = add([in0, lay_res], name="output_layer")
        else:
            lay_out = Conv2D(
                output_chan,
                (1, 1),
                activation="linear",
                name="regression_layer",
                kernel_initializer=conv_init,
            )(x)
    elif output_chan == 1:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="sigmoid",
            name="output_layer",
            kernel_initializer=conv_init,
        )(x)
    else:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="softmax",
            name="output_layer",
            kernel_initializer=conv_init,
        )(x)

    return Model(lay_input, lay_out)


# %% Parameterized 2.5D Block Model with 5 input slices
def BlockModelMS5(
    input_shape,
    filt_num=16,
    numBlocks=3,
    output_chan=1,
    regression=False,
    residual=False,
    use_bn=False,
    conv_init="glorot_uniform",
):
    """Creates a Block CED model for segmentation or image regression problems
    with 5 slice input and 1 slice output

    Args:
        input shape: a list or tuple of [rows,cols,slices,channels] of input images
        filt_num: the number of filters in the first and last layers
            This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        regression: Whether to have a continuous output with linear activation
        residual: Whether to create a ResNet where the network learns a residual to add
            to the input. The output of the network is added to the first channel of the
            middle slice of the input to generate the final output.
        use_bn: Whether to use batch normalization. Not recommended for GAN useage
        conv_init: The initializer string to use for convolution kernel initialization.


    Returns:
        An unintialized Keras model

    Example useage: GenModel = BlockModelMS5([256,256,5,1],filt_num=8)
        or try: GenModel = BlockModelMS5(x_train.shape[1:])

    Notes: Using rows/cols that are powers of 2 is recommended. Or,
        the rows/cols must be divisible by ^*numBlocks for skip connections
        to match up properly

    """

    # check for input shape compatibility
    rows, cols, slices = input_shape[0:3]
    assert slices == 5, "Number of slices must be 5"
    assert rows % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.maximum(rows, cols)
    minsize = (startsize - np.sum(2 ** np.arange(1, numBlocks + 1))) / 2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name="input_layer")

    MSpad = ZeroPadding3D(padding=(1, 1, 0), name="MSprepad1")(lay_input)
    MSconv = Conv3D(
        filt_num,
        (3, 3, 3),
        padding="valid",
        name="MSconv1",
        kernel_initializer=conv_init,
    )(MSpad)
    if use_bn:
        bn = batchnorm()(MSconv)
        MSact = ELU(name="MSelu1")(bn)
    else:
        MSact = ELU(name="MSelu1")(MSconv)

    MSpad = ZeroPadding3D(padding=(1, 1, 0), name="MSprepad2")(MSact)
    MSconv = Conv3D(
        filt_num,
        (3, 3, 3),
        padding="valid",
        name="MSconv2",
        kernel_initializer=conv_init,
    )(MSpad)
    if use_bn:
        bn = batchnorm()(MSconv)
        MSact = ELU(name="MSelu2")(bn)
    else:
        MSact = ELU(name="MSelu2")(MSconv)

    x = Reshape((rows, cols, filt_num))(MSact)

    # contracting blocks
    skip_list = []
    for rr in range(1, numBlocks + 1):
        x1 = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="same",
            name="Conv1_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_x1_{}".format(rr))(x1)
        x3 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv3_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_x3_{}".format(rr))(x3)
        x51 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv51_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_x51_{}".format(rr))(x51)
        x52 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv52_{}".format(rr),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_x52_{}".format(rr))(x52)
        x = concatenate([x1, x3, x52], name="merge_{}".format(rr))
        x = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="valid",
            name="ConvAll_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_all_{}".format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name="PrePad_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (4, 4),
            padding="valid",
            strides=(2, 2),
            name="DownSample_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_downsample_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="ConvClean_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_clean_{}".format(rr))(x)
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks + 1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x1 = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="same",
            name="DeConv1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_Dx1_{}".format(dd))(x1)
        x3 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv3_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_Dx3_{}".format(dd))(x3)
        x51 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv51_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_Dx51_{}".format(dd))(x51)
        x52 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv52_{}".format(dd),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_Dx52_{}".format(dd))(x52)
        x = concatenate([x1, x3, x52], name="Dmerge_{}".format(dd))
        x = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="valid",
            name="DeConvAll_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dall_{}".format(dd))(x)
        x = UpSampling2D(size=(2, 2), name="UpSample_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean1_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean2_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean2_{}".format(dd))(x)

    # classifier
    if regression:
        if residual:
            lay_res = Conv2D(
                1,
                (1, 1),
                activation="linear",
                name="residual_layer",
                kernel_initializer=conv_init,
            )(x)
            in0 = Lambda(lambda x: x[..., 2, 0], name="channel_split")(lay_input)
            in0 = Reshape([rows, cols, 1])(in0)
            lay_out = add([in0, lay_res], name="output_layer")
        else:
            lay_out = Conv2D(
                output_chan,
                (1, 1),
                activation="linear",
                name="regression_layer",
                kernel_initializer=conv_init,
            )(x)
    elif output_chan == 1:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="sigmoid",
            name="output_layer",
            kernel_initializer=conv_init,
        )(x)
    else:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="softmax",
            name="output_layer",
            kernel_initializer=conv_init,
        )(x)

    return Model(lay_input, lay_out)


# %% Generalized Unet Model
def Unet(input_shape, filt_num=16, numBlocks=3, output_chan=1, regression=False):
    lay_input = Input(shape=input_shape, name="input_layer")
    x = lay_input
    act_list = []
    # contracting blocks 1-n
    for rr in range(1, numBlocks + 1):
        x = Conv2D(filt_num * 2 ** (rr - 1), (3, 3), padding="same", name="Conv1_{}".format(rr))(x)
        x = ELU(name="elu{}_1".format(rr))(x)
        x = Conv2D(filt_num * 2 ** (rr - 1), (3, 3), padding="same", name="Conv2_{}".format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu{}_2".format(rr))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        act_list.append(x)

    # expanding block n
    dd = numBlocks
    fn = np.round(filt_num * 2 ** (dd - 2)).astype(np.int)
    x = Conv2D(fn, (3, 3), padding="same", name="DeConv1_{}".format(dd))(x)
    x = BatchNormalization()(x)
    x = ELU(name="elu_1d{}".format(dd))(x)
    x = Conv2D(fn, (3, 3), padding="same", name="DeConv2_{}".format(dd))(x)
    x = BatchNormalization()(x)
    x = ELU(name="elu_2d{}".format(dd))(x)
    x = UpSampling2D()(x)

    # expanding blocks n-1
    expnums = list(range(1, numBlocks))
    expnums.reverse()
    for dd in expnums:
        fn = np.round(filt_num * 2 ** (dd - 2)).astype(np.int)
        x = concatenate([act_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x = Conv2D(fn, (3, 3), padding="same", name="DeConv1_{}".format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_1d{}".format(dd))(x)
        x = Conv2D(fn, (3, 3), padding="same", name="DeConv2_{}".format(dd))(x)
        x = BatchNormalization()(x)
        x = ELU(name="elu_2d{}".format(dd))(x)
        x = UpSampling2D()(x)

    x = Conv2D(filt_num, (3, 3), padding="same", name="CleanUp_1")(x)
    x = Conv2D(filt_num, (3, 3), padding="same", name="CleanUp_2")(x)
    # classifier
    if regression:
        lay_out = Conv2D(output_chan, (1, 1), activation="linear", name="regression_layer")(x)
    elif output_chan == 1:
        lay_out = Conv2D(output_chan, (1, 1), activation="sigmoid", name="output_layer")(x)
    else:
        lay_out = Conv2D(output_chan, (1, 1), activation="softmax", name="output_layer")(x)

    return Model(lay_input, lay_out)


# %% 2D pix2pix discriminator
def Discriminator2D(conditional_input_shape, test_input_shape, filtnum=16):
    # Conditional Inputs
    lay_cond_input = Input(shape=conditional_input_shape, name="conditional_input")

    xcond = Conv2D(
        filtnum,
        (3, 3),
        padding="valid",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        name="FirstCondLayer",
    )(lay_cond_input)
    xcond = LeakyReLU(alpha=0.2, name="leaky_cond")(xcond)

    usebias = False

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}".format(rr),
    )(xcond)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}".format(rr),
    )(xcond)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}".format(rr),
    )(xcond)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}".format(rr),
    )(lay_act)
    lay_act1 = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    # Testing Input block
    lay_test_input = Input(shape=test_input_shape, name="test_input")
    xtest = Conv2D(
        filtnum,
        (3, 3),
        padding="valid",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="FirstTestLayer",
    )(lay_test_input)
    xtest = LeakyReLU(alpha=0.2, name="leaky_test")(xtest)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}t".format(rr),
    )(xtest)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}t".format(rr),
    )(xtest)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}t".format(rr),
    )(xtest)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}t".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}t".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}t".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1t".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}t".format(rr),
    )(lay_act)
    lay_act2 = LeakyReLU(alpha=0.2, name="leaky{}_2t".format(rr))(lay_stride)

    # Merge blocks
    lay_act = concatenate([lay_act1, lay_act2], name="InputMerge")
    # contracting blocks 2-5
    for rr in range(2, 6):
        lay_conv1 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv1_{}".format(rr),
        )(lay_act)
        lay_conv3 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv3_{}".format(rr),
        )(lay_act)
        lay_conv51 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv51_{}".format(rr),
        )(lay_act)
        lay_conv52 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv52_{}".format(rr),
        )(lay_conv51)
        lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
        lay_conv_all = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvAll_{}".format(rr),
        )(lay_merge)
        #        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
        lay_stride = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (4, 4),
            padding="valid",
            strides=(2, 2),
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvStride_{}".format(rr),
        )(lay_act)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1, kernel_initializer=conv_initD, name="Dense1")(lay_flat)

    return Model(inputs=[lay_cond_input, lay_test_input], outputs=[lay_dense])


# %% 3 multislice pix2pix discriminator
def DiscriminatorMS3(conditional_input_shape, test_input_shape, filtnum=16):
    # Conditional Inputs
    lay_cond_input = Input(shape=conditional_input_shape, name="conditional_input")

    MSconv = Conv3D(8, (3, 3, 3), padding="valid", name="MSconv")(lay_cond_input)
    MSact = LeakyReLU(name="MSleaky")(MSconv)
    x = Reshape((254, 254, 8))(MSact)

    xcond = Conv2D(
        filtnum,
        (3, 3),
        padding="same",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        name="FirstCondLayer",
    )(x)
    xcond = LeakyReLU(alpha=0.2, name="leaky_cond")(xcond)

    usebias = False

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}".format(rr),
    )(xcond)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}".format(rr),
    )(xcond)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}".format(rr),
    )(xcond)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}".format(rr),
    )(lay_act)
    lay_act1 = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    # Testing Input block
    lay_test_input = Input(shape=test_input_shape, name="test_input")
    xtest = Conv2D(
        filtnum,
        (3, 3),
        padding="valid",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="FirstTestLayer",
    )(lay_test_input)
    xtest = LeakyReLU(alpha=0.2, name="leaky_test")(xtest)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}t".format(rr),
    )(xtest)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}t".format(rr),
    )(xtest)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}t".format(rr),
    )(xtest)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}t".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}t".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}t".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1t".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}t".format(rr),
    )(lay_act)
    lay_act2 = LeakyReLU(alpha=0.2, name="leaky{}_2t".format(rr))(lay_stride)

    # Merge blocks
    lay_act = concatenate([lay_act1, lay_act2], name="InputMerge")
    # contracting blocks 2-5
    for rr in range(2, 6):
        lay_conv1 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv1_{}".format(rr),
        )(lay_act)
        lay_conv3 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv3_{}".format(rr),
        )(lay_act)
        lay_conv51 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv51_{}".format(rr),
        )(lay_act)
        lay_conv52 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv52_{}".format(rr),
        )(lay_conv51)
        lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
        lay_conv_all = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvAll_{}".format(rr),
        )(lay_merge)
        #        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
        lay_stride = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (4, 4),
            padding="valid",
            strides=(2, 2),
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvStride_{}".format(rr),
        )(lay_act)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1, kernel_initializer=conv_initD, name="Dense1")(lay_flat)

    return Model(inputs=[lay_cond_input, lay_test_input], outputs=[lay_dense])


# %% 5 multislice pix2pix discriminator
def DiscriminatorMS5(conditional_input_shape, test_input_shape, filtnum=16):
    # Conditional Inputs
    lay_cond_input = Input(shape=conditional_input_shape, name="conditional_input")

    MSconv = Conv3D(8, (3, 3, 3), padding="valid", name="MSconv1")(lay_cond_input)
    MSact = LeakyReLU(name="MSleaky1")(MSconv)
    MSconv = Conv3D(8, (3, 3, 3), padding="valid", name="MSconv2")(MSact)
    MSact = LeakyReLU(name="MSleaky2")(MSconv)
    MSconvRS = Reshape((252, 252, 8))(MSact)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), data_format=None)(MSconvRS)

    xcond = Conv2D(
        filtnum,
        (3, 3),
        padding="same",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        name="FirstCondLayer",
    )(x)
    xcond = LeakyReLU(alpha=0.2, name="leaky_cond")(xcond)

    usebias = False

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}".format(rr),
    )(xcond)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}".format(rr),
    )(xcond)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}".format(rr),
    )(xcond)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}".format(rr),
    )(lay_act)
    lay_act1 = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    # Testing Input block
    lay_test_input = Input(shape=test_input_shape, name="test_input")
    xtest = Conv2D(
        filtnum,
        (3, 3),
        padding="valid",
        strides=(1, 1),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="FirstTestLayer",
    )(lay_test_input)
    xtest = LeakyReLU(alpha=0.2, name="leaky_test")(xtest)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}t".format(rr),
    )(xtest)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}t".format(rr),
    )(xtest)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}t".format(rr),
    )(xtest)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}t".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}t".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}t".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1t".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}t".format(rr),
    )(lay_act)
    lay_act2 = LeakyReLU(alpha=0.2, name="leaky{}_2t".format(rr))(lay_stride)

    # Merge blocks
    lay_act = concatenate([lay_act1, lay_act2], name="InputMerge")
    # contracting blocks 2-5
    for rr in range(2, 6):
        lay_conv1 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv1_{}".format(rr),
        )(lay_act)
        lay_conv3 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv3_{}".format(rr),
        )(lay_act)
        lay_conv51 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv51_{}".format(rr),
        )(lay_act)
        lay_conv52 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv52_{}".format(rr),
        )(lay_conv51)
        lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
        lay_conv_all = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvAll_{}".format(rr),
        )(lay_merge)
        #        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
        lay_stride = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (4, 4),
            padding="valid",
            strides=(2, 2),
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvStride_{}".format(rr),
        )(lay_act)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1, kernel_initializer=conv_initD, name="Dense1")(lay_flat)

    return Model(inputs=[lay_cond_input, lay_test_input], outputs=[lay_dense])


# %% Cycle GAN Generator Model with limited receptive field
def CycleGANgenerator(
    input_shape,
    output_chan,
    filtnum=16,
    numBlocks=4,
    noStride=2,
    use_bn=False,
    reg=True,
    conv_initG="glorot_uniform",
):
    # arguments are input shape [x,y,channels] (no # slices)
    # and number of output channels

    # Create input layer
    lay_input = Input(shape=input_shape, name="input_layer")

    # numBlocks= number of "inception" blocks

    # noStride:
    # number of blocks to have strided convolution
    # blocks after this number will not be strided
    # This is to limit the generator's receptive field
    # set noStride=numB to use standard generator

    # filtnum= filter parameterization. Filter numbers grow linearly with
    # depth of net

    # Adjust this based on input image size if not 256x256
    padamt = 1

    # Cropping so that skip connections work out
    lay_crop = Cropping2D(((padamt, padamt), (padamt, padamt)))(lay_input)

    # contracting block 1
    rr = 1
    x1 = Conv2D(
        filtnum * rr,
        (1, 1),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv1_{}".format(rr),
    )(lay_crop)
    x1 = ELU(name="elu{}_1".format(rr))(x1)
    x3 = Conv2D(
        filtnum * rr,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv3_{}".format(rr),
    )(lay_crop)
    x3 = ELU(name="elu{}_3".format(rr))(x3)
    x51 = Conv2D(
        filtnum * rr,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv51_{}".format(rr),
    )(lay_crop)
    x51 = ELU(name="elu{}_51".format(rr))(x51)
    x52 = Conv2D(
        filtnum * rr,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv52_{}".format(rr),
    )(x51)
    x52 = ELU(name="elu{}_52".format(rr))(x52)
    lay_merge = concatenate([x1, x3, x52], name="merge_{}".format(rr))
    x = Conv2D(
        filtnum * rr,
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initG,
        use_bias=False,
        name="ConvAll_{}".format(rr),
    )(lay_merge)
    if use_bn:
        x = batchnorm()(x, training=1)
    x = ELU(name="elu{}_all".format(rr))(x)
    x = Conv2D(
        filtnum * rr,
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initG,
        name="ConvStride_{}".format(rr),
    )(x)
    x = ELU(name="elu{}_stride".format(rr))(x)
    act_list = [x]

    # contracting blocks 2->numB
    for rr in range(2, numBlocks + 1):
        x1 = Conv2D(
            filtnum * rr,
            (1, 1),
            padding="same",
            kernel_initializer=conv_initG,
            name="Conv1_{}".format(rr),
        )(x)
        x1 = ELU(name="elu{}_1".format(rr))(x1)
        x3 = Conv2D(
            filtnum * rr,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="Conv3_{}".format(rr),
        )(x)
        x3 = ELU(name="elu{}_3".format(rr))(x3)
        x51 = Conv2D(
            filtnum * rr,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="Conv51_{}".format(rr),
        )(x)
        x51 = ELU(name="elu{}_51".format(rr))(x51)
        x52 = Conv2D(
            filtnum * rr,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="Conv52_{}".format(rr),
        )(x51)
        x52 = ELU(name="elu{}_52".format(rr))(x52)
        x = concatenate([x1, x3, x52], name="merge_{}".format(rr))
        x = Conv2D(
            filtnum * rr,
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initG,
            use_bias=False,
            name="ConvAll_{}".format(rr),
        )(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu{}_all".format(rr))(x)
        if rr > noStride:
            x = Conv2D(
                filtnum * rr,
                (3, 3),
                padding="valid",
                strides=(1, 1),
                kernel_initializer=conv_initG,
                name="ConvNoStride_{}".format(rr),
            )(x)
        else:
            x = Conv2D(
                filtnum * rr,
                (4, 4),
                padding="valid",
                strides=(2, 2),
                kernel_initializer=conv_initG,
                name="ConvStride_{}".format(rr),
            )(x)
        x = ELU(name="elu{}_stride".format(rr))(x)
        act_list.append(x)

    # expanding block numB
    dd = numBlocks
    x1 = Conv2D(
        filtnum * dd,
        (1, 1),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv1_{}".format(dd),
    )(x)
    x1 = ELU(name="elu{}d_1".format(dd))(x1)
    x3 = Conv2D(
        filtnum * dd,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv3_{}".format(dd),
    )(x)
    x3 = ELU(name="elu{}d_3".format(dd))(x3)
    x51 = Conv2D(
        filtnum * dd,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv51_{}".format(dd),
    )(x)
    x51 = ELU(name="elu{}d_51".format(dd))(x51)
    x52 = Conv2D(
        filtnum * dd,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv52_{}".format(dd),
    )(x51)
    x52 = ELU(name="elu{}d_52".format(dd))(x52)
    x = concatenate([x1, x3, x52], name="merge_d{}".format(dd))
    x = Conv2D(
        filtnum * dd,
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initG,
        use_bias=False,
        name="DeConvAll_{}".format(dd),
    )(x)
    if dd > noStride:
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu{}d_all".format(dd))(x)
        x = Conv2DTranspose(
            filtnum * dd,
            (3, 3),
            kernel_initializer=conv_initG,
            use_bias=False,
            name="cleanup{}_1".format(dd),
        )(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu_cleanup{}_1".format(dd))(x)
        x = Conv2D(
            filtnum * dd,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            use_bias=False,
            name="cleanup{}_2".format(dd),
        )(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu_cleanup{}_2".format(dd))(x)
    else:
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu{}d_all".format(dd))(x)
        x = UpSampling2D()(x)
        x = Conv2DTranspose(
            filtnum * dd,
            (3, 3),
            kernel_initializer=conv_initG,
            use_bias=False,
            name="cleanup{}_1".format(dd),
        )(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu_cleanup{}_1".format(dd))(x)
        x = Conv2D(
            filtnum * dd,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            use_bias=False,
            name="cleanup{}_2".format(dd),
        )(x)
        if use_bn:
            x = batchnorm()(x, training=1)
        x = ELU(name="elu_cleanup{}_2".format(dd))(x)

    # expanding blocks (numB-1)->1
    expnums = list(range(1, numBlocks))
    expnums.reverse()
    for dd in expnums:
        x = concatenate([act_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x1 = Conv2D(
            filtnum * dd,
            (1, 1),
            padding="same",
            kernel_initializer=conv_initG,
            name="DeConv1_{}".format(dd),
        )(x)
        x1 = ELU(name="elu{}d_1".format(dd))(x1)
        x3 = Conv2D(
            filtnum * dd,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="DeConv3_{}".format(dd),
        )(x)
        x3 = ELU(name="elu{}d_3".format(dd))(x3)
        x51 = Conv2D(
            filtnum * dd,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="DeConv51_{}".format(dd),
        )(x)
        x51 = ELU(name="elu{}d_51".format(dd))(x51)
        x52 = Conv2D(
            filtnum * dd,
            (3, 3),
            padding="same",
            kernel_initializer=conv_initG,
            name="DeConv52_{}".format(dd),
        )(x51)
        x52 = ELU(name="elu{}d_52".format(dd))(x52)
        x = concatenate([x1, x3, x52], name="merge_d{}".format(dd))
        x = Conv2D(
            filtnum * dd,
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initG,
            use_bias=False,
            name="DeConvAll_{}".format(dd),
        )(x)
        if dd > noStride:
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name="elu{}d_all".format(dd))(x)
            x = Conv2DTranspose(
                filtnum * dd,
                (3, 3),
                kernel_initializer=conv_initG,
                use_bias=False,
                name="cleanup{}_1".format(dd),
            )(x)
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name="elu_cleanup{}_1".format(dd))(x)
            x = Conv2D(
                filtnum * dd,
                (3, 3),
                padding="same",
                kernel_initializer=conv_initG,
                use_bias=False,
                name="cleanup{}_2".format(dd),
            )(x)
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name="elu_cleanup{}_2".format(dd))(x)
        else:
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name="elu{}d_all".format(dd))(x)
            x = UpSampling2D()(x)
            x = Conv2DTranspose(
                filtnum * dd,
                (3, 3),
                kernel_initializer=conv_initG,
                use_bias=False,
                name="cleanup{}_1".format(dd),
            )(x)
            if use_bn:
                x = batchnorm()(x, training=1)
            x = ELU(name="elu_cleanup{}_1".format(dd))(x)
            x = Conv2D(
                filtnum * dd,
                (3, 3),
                padding="same",
                kernel_initializer=conv_initG,
                use_bias=False,
                name="cleanup{}_2".format(dd),
            )(x)

    # regressor or classifier
    # pad back to original size
    x = ZeroPadding2D(padding=((padamt, padamt), (padamt, padamt)), data_format=None)(x)
    if reg:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="linear",
            kernel_initializer=conv_initG,
            name="regression",
        )(x)
    else:
        lay_out = Conv2D(
            output_chan,
            (1, 1),
            activation="softmax",
            kernel_initializer=conv_initG,
            name="classification",
        )(x)

    return Model(lay_input, lay_out)


# %% Cycle GAN discriminator
def CycleGANdiscriminator(input_shape, filtnum=16, numBlocks=3):
    conv_initD = "he_normal"
    # Input same as generator- [x,y,channels]
    lay_input = Input(shape=input_shape, name="input")

    usebias = False
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv1_{}".format(rr),
    )(lay_input)
    lay_conv3 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv3_{}".format(rr),
    )(lay_input)
    lay_conv51 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv51_{}".format(rr),
    )(lay_input)
    lay_conv52 = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (3, 3),
        padding="same",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="Conv52_{}".format(rr),
    )(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
    lay_conv_all = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvAll_{}".format(rr),
    )(lay_merge)
    #    bn = batchnorm()(lay_conv_all, training=1)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
    lay_stride = Conv2D(
        filtnum * (2 ** (rr - 1)),
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initD,
        use_bias=usebias,
        name="ConvStride_{}".format(rr),
    )(lay_act)
    lay_act = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    # contracting blocks 2-numBlocks
    for rr in range(2, numBlocks + 1):
        lay_conv1 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv1_{}".format(rr),
        )(lay_act)
        lay_conv3 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv3_{}".format(rr),
        )(lay_act)
        lay_conv51 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv51_{}".format(rr),
        )(lay_act)
        lay_conv52 = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (3, 3),
            padding="same",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="Conv52_{}".format(rr),
        )(lay_conv51)
        lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52], name="merge_{}".format(rr))
        lay_conv_all = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (1, 1),
            padding="valid",
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvAll_{}".format(rr),
        )(lay_merge)
        #        bn = batchnorm()(lay_conv_all, training=1)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_1".format(rr))(lay_conv_all)
        lay_stride = Conv2D(
            filtnum * (2 ** (rr - 1)),
            (4, 4),
            padding="valid",
            strides=(2, 2),
            kernel_initializer=conv_initD,
            use_bias=usebias,
            name="ConvStride_{}".format(rr),
        )(lay_act)
        lay_act = LeakyReLU(alpha=0.2, name="leaky{}_2".format(rr))(lay_stride)

    lay_one = Conv2D(1, (3, 3), kernel_initializer=conv_initD, use_bias=usebias, name="ConvOne")(lay_act)
    lay_avg = GlobalAveragePooling2D()(lay_one)

    return Model(lay_input, lay_avg)


# %% DeepMuMap Model
def BlockModel_dual(
    input_shape,
    filt_num=16,
    numBlocks=3,
    output_chan=1,
    dual_output=True,
    use_bn=True,
    conv_init="glorot_uniform",
):
    """Creates a Block CED model for simultaneous segmentation and
    regression with 3 slice input and two 1 slice outputs

    Args:
        input shape: a list or tuple of [rows,cols,slices,channels] of input images
        filt_num: the number of filters in the first and last layers
            This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model
        output_chan: number of output channels. Set if doing multi-class segmentation
        dual_output: whether to have both regression and segmentation output. Only
            regression if False
        use_bn: Whether to use batch normalization. Not recommended for GAN useage
        conv_init: The initializer string to use for convolution kernel initialization.


    Returns:
        An unintialized Keras model

    Example useage: MuMapModel = BlockModel_dual([256,256,5,1],filt_num=8)
        or try: MuMapModel = BlockModel_dual(x_train.shape[1:])

    Notes: Using rows/cols that are powers of 2 is recommended. Or,
        the rows/cols must be divisible by 2**numBlocks for skip connections
        to match up properly

    """

    # check for input shape compatibility
    rows, cols, slices = input_shape[0:3]
    assert slices == 3, "Number of slices must be 3"
    assert rows % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"
    assert cols % 2**numBlocks == 0, "Input shape and number of blocks are incompatible"

    # calculate size reduction
    startsize = np.maximum(rows, cols)
    minsize = (startsize - np.sum(2 ** np.arange(1, numBlocks + 1))) / 2**numBlocks
    assert minsize > 4, "Too small of input for this many blocks. Use fewer blocks or larger input"

    # input layer
    lay_input = Input(shape=input_shape, name="input_layer")

    MSpad = ZeroPadding3D(padding=(1, 1, 0), name="MSprepad")(lay_input)
    MSconv = Conv3D(
        filt_num,
        (3, 3, 3),
        padding="valid",
        name="MSconv1",
        kernel_initializer=conv_init,
    )(MSpad)
    if use_bn:
        bn = batchnorm()(MSconv)
        MSact = ELU(name="MSelu1")(bn)
    else:
        MSact = ELU(name="MSelu1")(MSconv)

    x = Reshape((rows, cols, filt_num))(MSact)

    # contracting blocks
    skip_list = []
    for rr in range(1, numBlocks + 1):
        x1 = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="same",
            name="Conv1_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_x1_{}".format(rr))(x1)
        x3 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv3_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_x3_{}".format(rr))(x3)
        x51 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv51_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_x51_{}".format(rr))(x51)
        x52 = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="Conv52_{}".format(rr),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_x52_{}".format(rr))(x52)
        x = concatenate([x1, x3, x52], name="merge_{}".format(rr))
        x = Conv2D(
            filt_num * rr,
            (1, 1),
            padding="valid",
            name="ConvAll_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_all_{}".format(rr))(x)
        x = ZeroPadding2D(padding=(1, 1), name="PrePad_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (4, 4),
            padding="valid",
            strides=(2, 2),
            name="DownSample_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_downsample_{}".format(rr))(x)
        x = Conv2D(
            filt_num * rr,
            (3, 3),
            padding="same",
            name="ConvClean_{}".format(rr),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_clean_{}".format(rr))(x)
        skip_list.append(x)

    # expanding blocks
    expnums = list(range(1, numBlocks + 1))
    expnums.reverse()
    for dd in expnums:
        if dd < len(skip_list):
            x = concatenate([skip_list[dd - 1], x], name="skip_connect_{}".format(dd))
        x1 = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="same",
            name="DeConv1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x1 = BatchNormalization()(x1)
        x1 = ELU(name="elu_Dx1_{}".format(dd))(x1)
        x3 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv3_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x3 = BatchNormalization()(x3)
        x3 = ELU(name="elu_Dx3_{}".format(dd))(x3)
        x51 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv51_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x51 = BatchNormalization()(x51)
        x51 = ELU(name="elu_Dx51_{}".format(dd))(x51)
        x52 = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConv52_{}".format(dd),
            kernel_initializer=conv_init,
        )(x51)
        if use_bn:
            x52 = BatchNormalization()(x52)
        x52 = ELU(name="elu_Dx52_{}".format(dd))(x52)
        x = concatenate([x1, x3, x52], name="Dmerge_{}".format(dd))
        x = Conv2D(
            filt_num * dd,
            (1, 1),
            padding="valid",
            name="DeConvAll_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dall_{}".format(dd))(x)
        x = UpSampling2D(size=(2, 2), name="UpSample_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean1_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean1_{}".format(dd))(x)
        x = Conv2D(
            filt_num * dd,
            (3, 3),
            padding="same",
            name="DeConvClean2_{}".format(dd),
            kernel_initializer=conv_init,
        )(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = ELU(name="elu_Dclean2_{}".format(dd))(x)

    # regressor
    lay_reg = Conv2D(1, (1, 1), activation="linear", name="reg_output")(x)
    if dual_output:
        # classifier
        lay_class = Conv2D(
            output_chan,
            (1, 1),
            activation="softmax",
            name="class_output",
            kernel_initializer=conv_init,
        )(x)
        returnModel = Model(inputs=[lay_input], outputs=[lay_reg, lay_class])
    else:
        returnModel = Model(lay_input, lay_reg)

    return returnModel


# %%
def EncoderBlock(ind, filtmult, x):
    conv_initG = "glorot_uniform"
    x1 = Conv2D(
        filtmult * ind,
        (1, 1),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv1_{}".format(ind),
    )(x)
    x1 = ELU(name="elu_1_{}".format(ind))(x1)
    x3 = Conv2D(
        filtmult * ind,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv3_{}".format(ind),
    )(x)
    x3 = ELU(name="elu_3_{}".format(ind))(x3)
    x51 = Conv2D(
        filtmult * ind,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv51_{}".format(ind),
    )(x)
    x51 = ELU(name="elu_51_{}".format(ind))(x51)
    x52 = Conv2D(
        filtmult * ind,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="Conv52_{}".format(ind),
    )(x51)
    x52 = ELU(name="elu_52_{}".format(ind))(x52)
    lay_merge = concatenate([x1, x3, x52], name="merge_{}".format(ind))
    x = Conv2D(
        filtmult * ind,
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initG,
        name="ConvAll_{}".format(ind),
    )(lay_merge)
    x = ELU(name="elu_all_{}".format(ind))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(
        filtmult * ind,
        (4, 4),
        padding="valid",
        strides=(2, 2),
        kernel_initializer=conv_initG,
        name="ConvStride_{}".format(ind),
    )(x)
    x = ELU(name="elu_stride_{}".format(ind))(x)
    return x


# %%
def DecoderBlock(ind, ind2, filtmult, x, is_res=False):
    conv_initG = "glorot_uniform"
    filts = filtmult * (ind - ind2 + 1)
    x1 = Conv2D(
        filts,
        (1, 1),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv1_{}-{}".format(ind, ind2),
    )(x)
    x1 = ELU(name="elu_d1_{}-{}".format(ind, ind2))(x1)
    x3 = Conv2D(
        filts,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv3_{}-{}".format(ind, ind2),
    )(x)
    x3 = ELU(name="elu_d3_{}-{}".format(ind, ind2))(x3)
    x51 = Conv2D(
        filts,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv51_{}-{}".format(ind, ind2),
    )(x)
    x51 = ELU(name="elu_d51_{}-{}".format(ind, ind2))(x51)
    x52 = Conv2D(
        filts,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        name="DeConv52_{}-{}".format(ind, ind2),
    )(x51)
    x52 = ELU(name="elu_d52_{}-{}".format(ind, ind2))(x52)
    x = concatenate([x1, x3, x52], name="merge_d{}-{}".format(ind, ind2))
    x = Conv2D(
        filts,
        (1, 1),
        padding="valid",
        kernel_initializer=conv_initG,
        use_bias=False,
        name="DeConvAll_{}-{}".format(ind, ind2),
    )(x)
    x = ELU(name="elu_d-all_{}-{}".format(ind, ind2))(x)
    x = UpSampling2D()(x)
    x = Conv2DTranspose(
        filts,
        (3, 3),
        padding="same",
        kernel_initializer=conv_initG,
        use_bias=False,
        name="cleanup_{}-{}".format(ind, ind2),
    )(x)
    x = ELU(name="elu_cleanup_{}-{}".format(ind, ind2))(x)
    if is_res:
        x = Conv2D(
            1,
            (1, 1),
            padding="same",
            kernel_initializer=conv_initG,
            use_bias=False,
            name="res_{}".format(ind),
        )(x)
    return x


# %% Layered Residual Model with MS = 3
def ResBlockModel(input_shape, numBlocks=3, filtnum=16):
    """Creates a Block CED model for multi-residual regression
        with 3 slice input and 1 slice output

    Args:
        input shape: a list or tuple of [rows,cols,slices,channels] of input images
        filt_num: the number of filters in the first and last layers
            This number is multipled linearly increased and decreased throughout the model
        numBlocks: number of processing blocks. The larger the number the deeper the model

    Returns:
        An unintialized Keras model

    Notes: Using rows/cols that are powers of 2 is recommended. Otherwise,
        the rows/cols must be divisible by 2^numBlocks for skip connections
        to match up properly

    """
    # Input layer
    lay_input = Input(shape=input_shape, name="input_layer")
    rows, cols = input_shape[:2]
    # extract PET image channel from center slice
    res = Lambda(lambda x: x[:, 1, ..., 0], name="channel_split")(lay_input)
    res = Reshape([cols, rows, 1])(res)

    # Pad input before 3D conv to get right output shape
    padamt = 1
    zp = ZeroPadding3D(padding=((0, 0), (padamt, padamt), (padamt, padamt)), data_format=None)(lay_input)
    MSconv = Conv3D(filtnum, (3, 3, 3), padding="valid", name="MSconv")(zp)
    if use_bn:
        bn = batchnorm()(MSconv, training=1)
        MSact = ELU(name="MSelu")(bn)
    else:
        MSact = ELU(name="MSelu")(MSconv)
    x = Reshape((rows, cols, filtnum))(MSact)

    block_ends = []
    aa = 0
    for rr in range(1, numBlocks + 1):
        # encoder
        x = EncoderBlock(rr, filtnum, x)
        block_ends.append(x)
        for dd in range(1, rr + 1):
            # decoder
            x = DecoderBlock(rr, dd, filtnum, x, dd == rr)
            if dd == rr:
                res = add([res, x])
                x = block_ends[aa]
            else:
                x = concatenate([x, block_ends[aa]], name="connect_{}".format(aa))
                aa += 1
                block_ends.append(x)

    return Model(lay_input, res)
