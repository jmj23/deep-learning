# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:32:18 2017

@author: jmj136
"""
import numpy as np

def makeM(theta_z=0,tx=0,ty=0,sk_x=0,sk_y=0,sc_x=0,sc_y=0,imsize = 256):
    
    Sz, Cz = np.sin(theta_z), np.cos(theta_z)
    
    # Rotation matrices, angles theta, translation tx, ty
    Hz = np.array([[Cz, -Sz, tx],
                   [Sz,  Cz, ty],
                   [0,  0, 1]])
    
    # Translation matrix to shift the image center to the origin
    T = np.array([[1, 0, -imsize / 2.],
                  [0, 1, -imsize / 2.],
                  [0, 0, 1]])
    
    # x/y skew matrix
    S = np.array([[sc_x,  0,   0],
                  [0,    sc_y, 0],
                  [sk_x, sk_y, 1]])
    
    M = S.dot(np.linalg.inv(T).dot(Hz).dot(T))
    return M