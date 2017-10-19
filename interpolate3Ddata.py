#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:38:51 2017

@author: jmj136
"""

from scipy.interpolate import RegularGridInterpolator
from numpy import linspace, zeros, array
V = x_train[0,...,0]
xg = range(V.shape[2])
yg = range(V.shape[1])
zg = range(V.shape[0])

fn = RegularGridInterpolator((zg,yg,xg), V)
xq = xg
yq = yg
zq = linspace(0,V.shape[0],2*V.shape[0])
interpV = fn()