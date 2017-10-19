# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:11:24 2017

@author: jmj136
"""
import sys
from PyQt5 import QtGui  # (the example applies equally well to PySide)
import pyqtgraph as pg
import numpy as np

pg.setConfigOptions(imageAxisOrder='row-major')

## Always start by initializing Qt (only once per application)
app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

## Define a top-level widget to hold everything
Q = QtGui.QWidget()

## Create some widgets to be placed inside
btn = QtGui.QPushButton('press me')
text = QtGui.QLineEdit('enter text')
listw = QtGui.QListWidget()
view = Q.addViewBox()
imagew = pg.ImageView()
imagew.setImage(volume2,autoRange=True)
#plotw = pg.PlotWidget(title='Three plot curves')
#x = np.arange(1000)
#y = np.random.normal(size=(3,1000))
#for i in range(3):
#    plotw.plot(x,y[i],pen=(i,3))

## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
Q.setLayout(layout)

# set stretches
layout.setColumnStretch(0,1)

## Add widgets to the layout in their proper positions
layout.addWidget(imagew, 0, 0, 3, 1)  # plot goes on left side, spanning 3 rows
layout.addWidget(btn, 0, 1)   # button goes in upper-right
layout.addWidget(text, 1, 1)   # text edit goes in middle-right
layout.addWidget(listw, 2, 1)  # list widget goes in bottom-right


## Display the widget as a new window
Q.showMaximized()

## Start the Qt event loop
app.exec_()