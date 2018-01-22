# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:32:21 2017

@author: jmj136
"""
import keras.backend as K
import tensorflow as tf
#import numpy as np

def jac_met(y_true, y_pred):
    Xr = K.round(y_pred)
    X2r = K.round(y_true)
    intersection = Xr*X2r
    union = K.maximum(Xr, X2r)
    intsum = K.sum(intersection)
    unsum = K.sum(union)
    jacc =intsum/unsum
    return jacc

def perc_error(y_true, y_pred):
    y_true_f = K.round(K.flatten(y_true))
    y_pred_f = K.round(K.flatten(y_pred))
    intersect = y_true_f * y_pred_f
    non_intersect = 1-intersect
    exTrue = K.sum(y_true_f * non_intersect)
    exPred = K.sum(y_pred_f * non_intersect)
    return 100*(exTrue + exPred + .01) / (K.sum(y_true_f)+.01)

def bin_met(y_true,y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

def dice_met(y_true, y_pred):
    y_int = K.round(y_pred[...,0])*y_true[...,0]
    return K.sum(y_int)

def bin_met_test(y_true, y_pred):
    y_pred = K.round(K.flatten(y_pred))

    y_true = K.round(K.flatten(y_true))
    return K.mean(K.equal(y_true, K.round(y_pred)))

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    dc = dice_coef(y_true,y_pred)
    return 1 - dc

def ssim_loss(y_true,y_pred):    
    patches_true = tf.extract_image_patches(y_true, [1,4,4,1],[1,4,4,1],[1,1,1,1],padding='VALID')
    patches_pred = tf.extract_image_patches(y_pred, [1,4,4,1],[1,4,4,1],[1,1,1,1],padding='VALID')
    
#    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
#    patches_pred = K.reshape(patches_pred,[-1,w,h,c1*c2*c3])
#    patches_true = K.reshape(patches_true,[-1,w,h,c1*c2*c3])
    
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    eps = 1e-9
    std_true = K.sqrt(var_true+eps)
    std_pred = K.sqrt(var_pred+eps)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom #no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)

def weighted_mse(y_true, y_pred):
    y_true = K.flatten( y_true )
    y_pred = K.flatten( y_pred )

    bone_mask = K.cast( K.greater( y_true, 1.0 ), 'float32' )
    air_mask =  K.cast( K.less( y_true, 0.02 ), 'float32' )
    soft1_mask = K.cast( K.less( y_true, 1 ), 'float32' )
    soft2_mask = K.cast( K.greater( y_true, 0.02 ), 'float32' )
    soft_mask = soft1_mask * soft2_mask
    
    bone_true = bone_mask * y_true
    bone_pred = bone_mask * y_pred
    
    air_true = air_mask * y_true
    air_pred = air_mask * y_pred
    
    soft_true = soft_mask * y_true
    soft_pred = soft_mask * y_pred
    
    bone_loss = K.mean(K.square(bone_true - bone_pred), axis=-1)
    air_loss = K.mean(K.square(air_true - air_pred), axis=-1)
    soft_loss = K.mean(K.square(soft_true - soft_pred), axis=-1)
    
    return 1.3*bone_loss + 1.5*air_loss + soft_loss

def weighted_mae(y_true, y_pred):
    y_true = K.flatten( y_true )
    y_pred = K.flatten( y_pred )

    tis_mask1 = K.cast( K.greater( y_true, 0.01 ), 'float32' )
    tis_mask2 = K.cast( K.less( y_true, 0.7 ), 'float32' )
    tis_mask = tis_mask1 * tis_mask2
    les_mask =  K.cast( K.greater(y_true,0.7), 'float32' )
    air_mask =  K.cast( K.less( y_true, 0.01 ), 'float32' )
    
    tis_true = tis_mask * y_true
    tis_pred = tis_mask * y_pred
    
    air_true = air_mask * y_true
    air_pred = air_mask * y_pred
    
    les_true = les_mask * y_true
    les_pred = les_mask * y_pred
    
    tis_loss = K.mean(K.abs(tis_true - tis_pred), axis=-1)
    air_loss = K.mean(K.abs(air_true - air_pred), axis=-1)
    les_loss = K.mean(K.abs(les_true - les_pred), axis=-1)
    
    return .5*air_loss + 1.5*tis_loss + 2 * les_loss

def dice_coef_multi(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
