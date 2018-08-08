#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:39:55 2018

@author: jmj136
"""
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
 # Get files
itatmis_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_SimResults_{}_CV*.txt'.format('Cardiac')
non_path = '/home/jmj136/deep-learning/ItATMIS2/Abstract/Results/ItATMIS_NoAug_SimResults_{}_CV*.txt'.format('Cardiac')
it_result_files = glob(itatmis_path)
non_result_files = glob(non_path)

# Create a figure instance
fig = plt.figure(None, figsize=(9, 6))
# Create an axes instance
ax = fig.add_subplot(111)

   
# get itatmis scores
it_scores = np.stack([np.loadtxt(f) for f in it_result_files])
it_x = np.arange(1,it_scores.shape[1]+1)
# tile x array for displaying as one scatter plot
it_x = np.tile(it_x,(it_scores.shape[0],1))

# get non-augmented itatmis scores
non_scores = np.stack([np.loadtxt(f)[:16] for f in non_result_files])
non_x = np.arange(1,17)
# tile x array for displaying as one scatter plot
non_x = np.tile(non_x,(non_scores.shape[0],1))

ax.scatter(it_x.flatten(),it_scores.flatten(),c='r',label='ItATMIS')
ax.scatter(non_x.flatten(),non_scores.flatten(),c='b',label='Non-Aug ItATMIS')
    
plt.title('Dice Score over Iterations for {} Task'.format('Cardiac'))
plt.xlabel('Number of subjects')
plt.ylabel('Dice')
plt.legend()
plt.ylim([0,1])
plt.xlim([0,21])
plt.xticks(range(2,21,2))