#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:22:53 2017

@author: jmj136
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
import time

# data setup
def ConvertXdata(x_data):
    # pre allocate
    newSampNum = np.round(x_data.shape[0]/64).astype(np.int)
    x_out = np.zeros((newSampNum,8*28,8*28))
    for ii in range(0,x_data.shape[0],64):
        samp = np.round(ii/64).astype(np.int)
        for jj in range(8):
            for kk in range(8):
                x1 = jj*28
                x2 = (jj+1)*28
                y1 = kk*28
                y2 = (kk+1)*28
                ind = jj*4 + kk
                x_out[samp,x1:x2,y1:y2] = x_data[ii+ind]
    return x_out[...,np.newaxis]

def ConvertYdata(x_data,y_data):
    seg = np.zeros(x_data.shape)
    for ii in range(0,seg.shape[0]):
        curim = x_data[ii]
        curim[curim>0.3] = np.argmax(y_data[ii])+1
        curim[curim<=0.3] = 0
        seg[ii,...] = curim
        
    newSampNum = np.round(x_data.shape[0]/64).astype(np.int)
    y_out = np.zeros((newSampNum,8*28,8*28))
    for ii in range(0,x_data.shape[0],64):
        samp = np.round(ii/64).astype(np.int)
        for jj in range(8):
            for kk in range(8):
                x1 = jj*28
                x2 = (jj+1)*28
                y1 = kk*28
                y2 = (kk+1)*28
                ind = jj*4 + kk
                y_out[samp,x1:x2,y1:y2] = seg[ii+ind]
    y_class = tf.one_hot(y_out,depth=11)
    return y_class

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_ims = mnist.train.images[:51200]
mnist_lbls = mnist.train.labels[:51200]
del mnist
mnist_ims = np.reshape(mnist_ims,[mnist_ims.shape[0],28,28])
x_data = ConvertXdata(mnist_ims)
y_tensor = ConvertYdata(mnist_ims,mnist_lbls)
sess = tf.InteractiveSession()
with tf.Session() as sess:
    y_data = y_tensor.eval()
    
#%% layer definitions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2dS(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='VALID')

def conv2dF(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def get_batch(xdata,ydata,num,bs):
    ind = num*bs % (xdata.shape[0]-bs)
    return [xdata[ind:ind+bs],ydata[ind:ind+bs]]

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

#%% Defining variables
# inputs
x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1],x_data.shape[2],x_data.shape[3]])
y_ = tf.placeholder(tf.float32, shape=[None, y_data.shape[1],y_data.shape[2],y_data.shape[3]])
  
# First layer
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

W_conv1s = weight_variable([4, 4, 16, 16])
b_conv1s = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_conv1s = tf.nn.relu(conv2dS(h_conv1, W_conv1s) + b_conv1s)
# Second layer
W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])

W_conv2s = weight_variable([4, 4, 32, 32])
b_conv2s = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_conv1s, W_conv2) + b_conv2)
h_conv2s = tf.nn.relu(conv2dS(h_conv2, W_conv2s) + b_conv2s)
# Third layer
W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])

W_conv3s = weight_variable([4, 4, 64, 64])
b_conv3s = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_conv2s, W_conv3) + b_conv3)
h_conv3s = tf.nn.relu(conv2dS(h_conv3, W_conv3s) + b_conv3s)
# Deconvolutional layers
W_deconv1s = weight_variable([3, 3, 64, 64])
b_deconv1s = bias_variable([64])

W_deconv2s = weight_variable([3, 3, 64, 64])
b_deconv2s = bias_variable([64])

W_deconv3s = weight_variable([3, 3, 64, 64])
b_deconv3s = bias_variable([64])

W_deconv4 = weight_variable([3,3,32,64])
b_deconv4 = bias_variable([32])

W_deconv5 = weight_variable([4, 4, 32, 16])
b_deconv5 = bias_variable([16])

dim = (h_conv3s.get_shape()[2].value)*2+2
dim2 = dim*2+2
dim3 = dim2*2+2
dim4 = dim3+2
h_deconv1s = tf.nn.relu(tf.nn.conv2d_transpose(h_conv3s,W_deconv1s,tf.constant([16,dim,dim,64]),strides=[1,2,2,1],padding='VALID') + b_deconv1s)
h_deconv2s = tf.nn.relu(tf.nn.conv2d_transpose(h_deconv1s,W_deconv2s,tf.constant([16,dim2,dim2,64]),strides=[1,2,2,1],padding='VALID') + b_deconv2s)
h_deconv3s = tf.nn.relu(tf.nn.conv2d_transpose(h_deconv2s,W_deconv3s,tf.constant([16,dim3,dim3,64]),strides=[1,2,2,1],padding='VALID')+b_deconv3s)
h_deconv4 = tf.nn.relu(tf.nn.conv2d_transpose(h_deconv3s,W_deconv4,tf.constant([16,dim4,dim4,32]),strides=[1,1,1,1],padding='VALID')+b_deconv4)
h_deconv5 = tf.nn.relu(conv2d(h_deconv4, W_deconv5) + b_deconv5)
# Final classification layer
W_convF = weight_variable([3, 3, 16, 11])
b_convF = bias_variable([11])
y_conv = conv2dF(h_deconv5,W_convF) + b_convF

prediction = tf.nn.softmax(y_conv,-1)
classification = tf.argmax(prediction,-1)
flat_logits = tf.reshape(y_conv, [-1, 11])
flat_labels = tf.reshape(y_, [-1, 11])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                              labels=flat_labels))

train_op=tf.train.AdamOptimizer(1e-4).minimize(loss)

time1 = time.time()
numSteps = 3000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(numSteps):
        batch = get_batch(x_data,y_data,i,16)
        if i % 100 == 0:
            train_loss = loss.eval(feed_dict={
                    x: batch[0], y_: batch[1]})
            print('step %d, loss: %g' % (i, train_loss))
        train_op.run(feed_dict={x: batch[0], y_: batch[1]})
    time2 = time.time()
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/home/jmj136/deep-learning/MNIST-seg-model.ckpt")
    
tps = 1000*(time2-time1)/numSteps
print('Time per step: {:.02f} ms'.format(tps))

#%%
with tf.Session() as sess:
  sess.run(tf.local_variables_initializer())
  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state("/home/jmj136/deep-learning/")
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      saver.restore(sess, ckpt.model_checkpoint_path)
  output = classification.eval(feed_dict={
          x: batch[0], y_: batch[1]})
    
from VisTools import multi_slice_viewer0
multi_slice_viewer0(np.c_[(output)/10,(np.argmax(batch[1],axis=3))/10])