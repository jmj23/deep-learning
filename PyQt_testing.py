import sys
from PyQt5 import QtWidgets, uic
import scipy.io as spio
import pyqtgraph as pg
import numpy as np


pyqtd_main= "pyqt_main.ui"
pyqtd_imp = "pyqt_import.ui"

Ui_MainWindow, QtBaseClass1 = uic.loadUiType(pyqtd_main)
Ui_ImpWindow, QtBaseClass2 = uic.loadUiType(pyqtd_imp)
 
class MainApp(QtBaseClass1, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionImport_Images.triggered.connect(self.Import)
        self.ui.actionExit.triggered.connect(self.closeEvent)
        
        
        
    def Import(self):
        self.ImpW = ImpWindow(self)
        self.ImpW.show()
    def InitDisplay(self):
        ## add View Box to graphics view
        self.vbox = self.ui.graphicsView.addViewBox(border='r',lockAspect=True,
                                               enableMenu = False,enableMouse=False)
        ## Create image item
        self.img_item = pg.ImageItem()
        self.vbox.addItem(self.img_item)
        
        # Add initial image
        imshape = self.images.shape
        self.img_item.setImage(self.images[:,:,np.round(imshape[2]/2)])
        
    def closeEvent(self, evnt):
        print('closing')
        self.destroy()
#%%
class ImpWindow(QtBaseClass2,Ui_ImpWindow):
    def __init__(self,parent):
        super(ImpWindow,self).__init__()
        self.ui = Ui_ImpWindow()
        self.ui.setupUi(self)
        
    def pick_mat(self):
        w = QtWidgets.QWidget()
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(w, 'Open File')
        self.mat = spio.loadmat(filename,squeeze_me=True)
        print(filename)

#%%
if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        print('rerunning')
    window = MainApp()
    window.show()
    app.exec_()