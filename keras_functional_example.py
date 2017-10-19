# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:39:19 2017

@author: jmj136
"""
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

data = np.random.random((100,100,100,3))

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64,activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs,outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels) # starts training
