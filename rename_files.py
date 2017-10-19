#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:08:22 2017

@author: jmj136
"""

import os
path = '/media/jmj136/JJ_DATA/Keras Scripts/MatData'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, "segdata{:03d}.mat".format(i)))
    i = i+1