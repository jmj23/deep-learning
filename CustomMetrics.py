# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:32:21 2017

@author: jmj136
"""
import keras.backend as K
#import tensorflow as tf

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