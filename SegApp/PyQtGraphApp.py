# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:47:25 2017

@author: jmj136
"""

import sys
import scipy.io as spio
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np

import pyqtgraph.ptime as ptime

 
class MyApp(QtGui.QMainWindow):
    def __init__(self,parent=None):
        super(MyApp, self).__init__(parent)
        ## Create window with GraphicsView widget
        win = pg.GraphicsLayoutWidget()
        win.show()  ## show widget alone in its own window
        win.setWindowTitle('pyqtgraph example: ImageItem')
        view = win.addViewBox()
        
        ## lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)
        
        # Create button
        self.button = QtWidgets.QPushButton()
        self.button.clicked.connect(self.start)
        ## Create image item
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)
        
        ## Set initial view bounds
        view.setRange(QtCore.QRectF(0, 0, 600, 600))
        
        ## Create random image
        self.data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
        self.i = 0
        
        self.updateTime = ptime.time()
        self.fps = 0
        
    def start(self):
        self.updateData()
        
    def updateData(self):
        ## Display the data
        self.img.setImage(self.data[self.i])
        self.i = (self.i+1) % self.data.shape[0]
    
        QtCore.QTimer.singleShot(1, self.updateData)
        now = ptime.time()
        fps2 = 1.0 / (now-self.updateTime)
        self.updateTime = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        
        print("%0.1f fps" % self.fps)

        
    def LoadImages(self):
        w = QtGui.QWidget()
        filename,_ = QtGui.QFileDialog.getOpenFileName(w, 'Open File')
        self.mat = spio.loadmat(filename,squeeze_me=True)
        print(filename)
        self.img = self.mat['circ_y'][:,:,0]
        self.ui.graphicsView.setImage(self.img)
        
    def closeEvent(self, evnt):
        self.timer.stop()
        
if __name__ == "__main__":
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('rerunning')
    window = MyApp()
    window.show()
    app.exec_()