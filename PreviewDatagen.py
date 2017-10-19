# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:09:42 2017

@author: jmj136
"""

i = 0
for batch in datagen.flow(x_train, batch_size=1,
                          save_to_dir='datagen_preview', save_prefix='datagen_test', save_format='jpeg'):
    i += 1
    if i > 20:
        break