import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import pyqtProperty,pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor
import pyqtgraph as pg
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.draw import polygon
import os
# Use first available GPU
import GPUtil
try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except RuntimeError as e:
    print('No GPU available')
import psutil
import configparser
import nibabel as nib
from natsort import natsorted
import time
import keras

pg.setConfigOptions(imageAxisOrder='row-major')

main_ui_file= "main.ui"
select_ui_file = "DataSelect.ui"

Ui_MainWindow, QtBaseClass1 = uic.loadUiType(main_ui_file)
Ui_DataSelect, QtBaseClass2 = uic.loadUiType(select_ui_file)

#%%
class MainApp(QtBaseClass1,Ui_MainWindow):
    def __init__(self):
        try:
            super(MainApp,self).__init__()
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
            
            self.ui.progBar.setVisible(False)
            
            # attach callbacks
#            self.ui.actionSaveModel.triggered.connect(self.saveCor)
            self.ui.actionReset_View.triggered.connect(self.resetView)
            self.ui.actionUndo.triggered.connect(self.undo)
            self.ui.pb_SelectData.clicked.connect(self.DataSelect)
            
            # initialize some variables
            # configuration file name
            self.configFN = 'config.ini'
            # list of files to be included in processing            
            self.file_list = []
            # current index in file list
            self.FNind = 0
            # most recently used directory of data
            self.datadir= []
            # current images loaded into app
            self.images = []
            # current mask for current images
            self.mask = []
            # prepared inputs to model
            self.inputs = []
            # deep learning model
            self.model = []
            # output from CNN
            self.segOutput = []
            # spatial resolution of images
            self.spatres = []
            # boolean for displaying mask
            self.maskdisp = False
            # whether model is saved
            self.saved = True
            # multiplier for window-leveling
            self.WLmult = .5
            # threshold for segmentation
            self.segThresh = .5
            # set intial alpha
            self.alph = .3
            self.ui.slideAlpha.setValue(10*self.alph)
            # initialize display message
            self.disp_msg = 'Initializing...'
            
            # Initialize or load config file
            if os.path.isfile(self.configFN):
                config = configparser.ConfigParser()
                try:
                    config.read(self.configFN);
                except Exception as e:
                    self.error_msg = 'Unable to read config file. Creating new'
                    config = configparser.ConfigParser()
                    curdir = os.getcwd()
                    config['DEFAULT'] = {'data directory': curdir}
                    with open(self.configFN, 'w') as configfile:
                        config.write(configfile)
            else:
                config = configparser.ConfigParser()
                curdir = os.getcwd()
                config['DEFAULT'] = {'data directory': curdir}
                with open(self.configFN, 'w') as configfile:
                    config.write(configfile)
            # parse config into self
            self.config = config
            self.datadir = self.config.get('DEFAULT','data directory')
                    
            self.disp_msg = 'Welcome to ItATMIS'
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def DataSelect(self):
        self.DataSelectW = DataSelect(self)
        self.DataSelectW.show()
        
    def ImportImages(self):
        self.disp_msg = "Importing images..."
        curFN = self.file_list[self.FNind]
        self.ui.progBar.setEnabled(True)
        self.ui.progBar.setRange(0,0)
        self.imp_thread = NiftiImportThread(curFN)
        self.imp_thread.finished.connect(self.imp_finish_imp)
        self.imp_thread.images_sig.connect(self.gotImages)
        self.imp_thread.errorsig.connect(self.impError)
        self.imp_thread.start()
    
    def imp_finish_imp(self):
        self.ui.progBar.setRange(0,2)
        self.disp_msg = 'Images Imported'
        self.saved = False
        self.InitDisplay()
    
    def gotImages(self,images):
        self.images = images
        self.volshape = images.shape
    
    def impError(self):
        self.error_msg = 'Error import images'
            
    def InitDisplay(self):
        try:
            # create empty segmask
            self.segmask = np.zeros(self.images.shape)
            
            # create empty display mask
            msksiz = np.r_[self.volshape,4]
            msk = np.zeros(msksiz,dtype=np.float)
            msk[...,0] = 1
            msk[...,1] = 1
            msk[...,2] = 0
            msk[...,3] = self.alph*self.segmask
            self.mask = msk
            
            
            # setup plots
            asprat1 = 1
            self.vbox = self.ui.viewAxial.addViewBox(border=None,
                                                    enableMenu=False,
                                                    enableMouse=False,
                                                    invertY=True)
            self.vbox.setAspectLocked(True,ratio=asprat1)
            self.img_item = pg.ImageItem()
            self.vbox.addItem(self.img_item)
            self.img_item.setBorder((255,0,0,100))
            self.msk_item = pg.ImageItem()
            self.vbox.addItem(self.msk_item)
            
            # Add initial image and mask
            ind = np.round(self.images.shape[0]/2).astype(np.int16)
            self.ind = ind
            self.img_item.setImage(self.images[self.ind,...])
            self.msk_item.setImage(self.mask[self.ind,...])
            self.curmask = self.segmask[self.ind,...]
            self.prev_mask = self.segmask[self.ind,...]
            #adjust view range
            self.vbox.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[1]),
                                  padding=0.,disableAutoRange=True)
            # calculate window/leveling multiplier
            self.WLmult = self.img_item.getLevels()[1]/500
            
            # create starting brush            
            self.BrushMult = 1
            self.brush = self.my_brush_mask(self.ui.slideBrushSize.value()*self.BrushMult)
            self.rad = self.ui.slideBrushSize.value()
            
            # create brush cursor
            self.dot = pg.ScatterPlotItem(x=np.array([5.]),y=np.array([5.]),
                                    symbol='o',symbolSize=2*self.rad+1,symbolPen=(100,100,100,.5),
                                    brush=None,pxMode=False)
            self.vbox.addItem(self.dot)
            pg.SignalProxy(self.ui.viewAxial.scene().sigMouseMoved, rateLimit=100, slot=self.mouseMoved)
            self.ui.viewAxial.scene().sigMouseMoved.connect(self.mouseMoved)
            
            # Setup slider
            self.ui.slideAx.setMaximum(self.images.shape[0]-1)
            self.ui.slideAx.setValue(self.ind)
            
            # attach call backs
            self.img_item.mousePressEvent = self.clickEvent
            self.img_item.hoverEvent = self.brushHover
            self.msk_item.mousePressEvent = self.clickEvent
            self.ui.slideAx.valueChanged.connect(self.slide)
            self.ui.slideAlpha.valueChanged.connect(self.alphaSlide)
            self.ui.slideBrushSize.valueChanged.connect(self.brushSlide)
            self.ui.viewAxial.wheelEvent = self.scroll
            self.ui.viewAxial.keyPressEvent = self.vbKeyPress
            
            # make full screen
#            self.showMaximized()
            scrnsiz = QtWidgets.QDesktopWidget().screenGeometry()
            cntr = scrnsiz.center()
            width = 1011
            height = 675
            xv = np.round(cntr.x()-width/2)
            yv = np.round(cntr.y()-height/2)
            sizRect = QtCore.QRect(xv,yv,width,height)
            self.setGeometry(sizRect)
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def slide(self,ev):
        try:
            self.img_item.setImage(self.images[ev,...],autoLevels=False)
            self.msk_item.setImage(self.mask[ev,...])
            self.ind = ev
            # update undo mask
            self.prev_mask = np.copy(self.segmask[ev,...])
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def scroll(self,event):
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods == QtCore.Qt.ControlModifier:
            bump = -event.angleDelta().y()/120
            fac = 1+.1*bump
            pos = self.vbox.mapToView(event.pos())
            self.vbox.scaleBy(s=fac,center=(pos.x(),pos.y()))
            event.accept()
        else:
            curval = self.ui.slideAx.value()
            newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                             self.volshape[0]-1))
            self.ui.slideAx.setValue(newval)
    def brushSlide(self,event):
        self.brush = self.my_brush_mask(event*self.BrushMult)
        self.rad = event
        try:
            self.dot.setSize(2*event+1)
            curx,cury = self.dot.getData()
            self.dot.setData(x=curx,y=cury)
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def vbKeyPress(self,ev):
        try:
            ev.accept()
            if ev.text() == ']':
                curval = self.ui.slideBrushSize.value()
                self.ui.slideBrushSize.setValue(curval+1)
            elif ev.text() == '[':
                curval = self.ui.slideBrushSize.value()
                self.ui.slideBrushSize.setValue(curval-1)
            else:
                ev.ignore()
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def my_brush_mask(self,r):
        y,x = np.ogrid[-r: r+1, -r: r+1]
        mask = x**2+y**2 <= r**2+.5*r
        return mask.astype(int)
    def alphaSlide(self,ev):
        self.alph = .1*ev
        self.updateMask()
    def mouseMoved(self,pos):
        if self.img_item.sceneBoundingRect().contains(pos):
            mousePoint = self.vbox.mapSceneToView(pos)
            self.dot.setData(x=np.array([mousePoint.x()]),y=np.array([mousePoint.y()]))
    def brushHover(self,ev):
        if ev.isEnter():
            QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
            self.dot.setData(symbolPen=(100,100,100,0.5))
        elif ev.isExit():
            QtWidgets.QApplication.restoreOverrideCursor()
            self.dot.setData(symbolPen=(100,100,100,0))
    def clickEvent(self,ev):
        if ev.button()==1 or ev.button()==2:
            self.img_item.mouseMoveEvent = self.movingEvent
            self.msk_item.mouseMoveEvent = self.movingEvent
            self.img_item.mouseReleaseEvent = self.releaseEvent
            self.msk_item.mouseReleaseEvent = self.releaseEvent
            self.curmask = self.segmask[self.ind,...]
            self.prev_mask = np.copy(self.curmask)
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.bt = ev.button()
            self.draw(posx,posy)
            ev.accept()
        elif ev.button()==4:
            self.img_item.mouseMoveEvent = self.levelEvent
            self.msk_item.mouseMoveEvent = self.levelEvent
            self.img_item.mouseReleaseEvent = self.releaseEvent
            self.msk_item.mouseReleaseEvent = self.releaseEvent
            self.prevLevel = np.array(self.img_item.getLevels())
            self.startPos = np.array([ev.pos().x(),ev.pos().y()])
            ev.accept()
        else:
            ev.ignore()
            
    def movingEvent(self,ev):
        posx = ev.pos().x()
        posy = ev.pos().y()
        self.draw(posx,posy)
        ev.accept()
        
    def releaseEvent(self,ev):
        self.img_item.mouseMoveEvent = ''
        self.msk_item.mouseMoveEvent = ''
        if ev.button()==1 or ev.button()==2:
            fmask = binary_fill_holes(self.curmask)
            self.segmask[self.ind,...] = fmask
            self.updateMask()
#            self.calcFunc()
        ev.accept()
        
    def draw(self,x,y):
        try:
            brush = self.brush
            cmask = self.curmask
            bt = self.bt
            m,n = cmask.shape
            lby,uby = np.int16(np.max((y-self.rad,0))),np.int16(np.min((y+self.rad,m))+1)
            lbx,ubx = np.int16(np.max((x-self.rad,0))),np.int16(np.min((x+self.rad,n))+1)
            reg = cmask[lby:uby,lbx:ubx]
            pbrush = np.copy(brush)
            if lby==0:
                ybump = uby-lby
                pbrush = pbrush[-ybump:,:]
            elif uby == m+1:
                ybump = uby-lby-1
                pbrush = pbrush[:ybump,:]
            if lbx==0:
                xbump = ubx-lbx
                pbrush = pbrush[:,-xbump:]
            elif ubx==n+1:
                xbump = ubx-lbx-1
                pbrush = pbrush[:,:xbump]
            if np.array_equal(reg.shape,pbrush.shape):
                if bt == 1:
                    reg = np.maximum(pbrush,reg);
                    cmask[lby:uby,lbx:ubx] = reg          
                else:
                    reg = np.minimum(1-np.minimum(pbrush,reg),reg)
                    cmask[lby:uby,lbx:ubx] = reg
                    
            self.curmask = cmask
            self.mask[self.ind,...,3] = self.alph*cmask
            self.msk_item.setImage(self.mask[self.ind,...])
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
    
    def updateMask(self):
        self.mask[...,3] = self.alph*self.segmask
        self.msk_item.setImage(self.mask[self.ind,...])
        
    def showUncorMask(self):
        if self.ui.buttonUncorMask.checked:
            self.uncormask[...,3] = self.alph*self.parent.uncor_segmask
        else:
            self.uncormask[...,3] = 0
        self.msk_uncor_item.setImage(self.mask_uncor[self.ind,...])
        
    def undo(self):
        try:
            temp = np.copy(self.prev_mask)
            self.prev_mask = np.copy(self.segmask[self.ind,...])
            self.segmask[self.ind,...] = temp
            self.updateMask()
            self.calcFunc()
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def calcFunc(self):
        try:
            self.calc_thread = CalcThread(self.parent.uncor_segmask,self.segmask)
            self.calc_thread.error_sig.connect(self.calcError)
            self.calc_thread.finished.connect(self.calcFinish)
            self.calc_thread.calcs_sig.connect(self.calcGotCalcs)
            
            self.calc_thread.start()
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
    def calcError(self,msg):
        self.parent.error_msg = msg
    
    def calcGotCalcs(self,calcs):
        self.ui.lblOTV.setText("{0:.1f} cc".format(calcs[0]))
        self.ui.lblCTV.setText("{0:.1f} cc".format(calcs[1]))
        self.ui.lblOUV.setText("{0:.1f} cc".format(calcs[2]))
        self.ui.lblCUV.setText("{0:.1f} cc".format(calcs[3]))
        self.ui.lblDiff.setText("{0:.2f} %".format(calcs[4]))
        self.ui.lblDice.setText("{0:.2f}".format(calcs[5]))
    def calcFinish(self):
        pass
        
    def levelEvent(self,ev):
        curpos = np.array([ev.pos().x(),ev.pos().y()])
        posdiff = self.WLmult*(curpos-self.startPos)
        prevLevels = self.prevLevel
        newW = prevLevels[1]-prevLevels[0]+posdiff[0]
        newL = np.mean(prevLevels)+posdiff[1]
        newLev0 = newL-newW/2
        newLev1 = newL+newW/2
        self.img_item.setLevels([newLev0,newLev1])
        ev.accept()
        
    def resetView(self,ev):
        self.vbox.setRange(xRange=(0,self.volshape[2]),
                            yRange=(0,self.volshape[1]))
        self.img_item.setImage(self.images[self.ind,...],autoLevels=True)
        
            
    def saveConfig(self):
        self.config['DEFAULT']['data directory'] = self.datadir
        with open(self.configFN, 'w') as configfile:
            self.config.write(configfile)
        
    @pyqtProperty(str)
    def disp_msg(self):
            return self.disp_msg
    @disp_msg.setter
    def disp_msg(self, value):
        self._disp_msg = value
        item = QtWidgets.QListWidgetItem(value)
        self.ui.listDisp.addItem(item)
        if self.ui.listDisp.count() > 200:
            self.ui.listDisp.takeItem(0)
        self.ui.listDisp.scrollToBottom()
        self.ui.listDisp.update()
        print(value)
        QtWidgets.qApp.processEvents()
        
    @pyqtProperty(str)
    def error_msg(self):
            return self._error_msg
    @error_msg.setter
    def error_msg(self, value):
        self._error_msg = value
        item = QtWidgets.QListWidgetItem(value)
        item.setForeground(QtCore.Qt.red)
        self.ui.listDisp.addItem(item)
        if self.ui.listDisp.count() > 200:
            self.ui.listDisp.takeItem(0)
        self.ui.listDisp.scrollToBottom()
        self.ui.listDisp.update()
        print('Error:',value)
        QtWidgets.qApp.processEvents()
        
    def closeEvent(self, ev):
        # check to quit
        quit_msg = "Are you sure you wish to close?"
        reply = QtWidgets.QMessageBox.warning(self, 'Closing', 
                         quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
    
        if reply == QtWidgets.QMessageBox.Yes:
            ev.accept()
        else:
            ev.ignore()
            return
        # save config file
        self.saveConfig()
        # exit
        del self


#%%
class NotMainApp(QtBaseClass1, Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.progBar.setVisible(False)
        # attach menus
        self.ui.actionImport_Images.triggered.connect(self.Import)
        self.ui.actionExit.triggered.connect(self.closeEvent)
        self.ui.actionLoad_Data.triggered.connect(self.data_load)
        self.ui.actionSave_Data.triggered.connect(self.data_save)
        self.ui.actionLoad_Model.triggered.connect(self.load_model)
        self.ui.actionLoad_Model_Weights.triggered.connect(self.load_model_weights)
        self.ui.actionEdit_Preferences.triggered.connect(self.setPreferences)
        
        
        self.ui.actionSegment.triggered.connect(self.segment)
        self.ui.actionCorrect_Segmentation.triggered.connect(self.openCorrector)
        self.ui.actionCalculate_PDWF.triggered.connect(self.calcPDWF)
        self.ui.actionReset_Mask.triggered.connect(self.resetMask)
        self.ui.actionShift_Dimensions.triggered.connect(self.shiftDims)
        self.ui.actionFlip_L_R.triggered.connect(lambda: self.flipIms(2))
        self.ui.actionFlip_A_P.triggered.connect(lambda: self.flipIms(1))
        self.ui.actionFlip_S_I.triggered.connect(lambda: self.flipIms(0))
        
        self.ui.actionOpen_User_Manual.triggered.connect(self.openHelp)
        
        # initialize some variables
        self.configFN = 'config.ini'
        self.CorW = []
        
        self.imagesW = []
        self.imagesF = []
        self.curimages = []
        self.curimagetype = 'water'
        self.PDWFmap = []
        self.PDWFthresh = 2
        self.inputs = []
        self.model = []
        self.segOutput = []
        self.segmask = []
        self.uncor_segmask = []
        self.SIbounds = np.array([])
        self.vpoint = np.array([])
        self.vpointL = np.array([])
        self.vpointR = np.array([])
        self.fov = np.array([22,36,36])
        self.spatres = []
        self.maskdisp = False
        self.bounddisp = False
        self.saved = True
        self.WLmult = .5
        self.mask_alpha = .4
        self.segThresh = .5
        self.cursel = []
        self.savename = ''
        self.prefImpType = 'dcm'
        
        
        # initialize display message
        self.disp_msg = 'Welcome to PDWFnet'
        
        # Initialize or load config file
        if os.path.isfile(self.configFN):
            config = configparser.ConfigParser()
            try:
                config.read(self.configFN);
            except Exception as e:
                self.error_msg = 'Unable to read config file. Creating new'
                config = configparser.ConfigParser()
                curdir = os.getcwd()
                fovstr = ",".join(str(e) for e in self.fov)
                alpha = str(self.mask_alpha)
                thresh = str(self.PDWFthresh)
                config['DEFAULT'] = {'MAT Directory': curdir,
                          'Import Directory': curdir,
                          'Model directory': curdir,
                          'Save Directory': curdir,
                          'FOV': fovstr,
                          'MaskAlpha': alpha,
                          'SegThresh': .5,
                          'PDWFthresh': thresh,
                          'Preferred Import Type': 'dcm'}
                with open(self.configFN, 'w') as configfile:
                    config.write(configfile)
        else:
            config = configparser.ConfigParser()
            curdir = os.getcwd()
            fovstr = ",".join(str(e) for e in self.fov)
            alpha = str(self.mask_alpha)
            thresh = str(self.PDWFthresh)
            config['DEFAULT'] = {'MAT Directory': curdir,
                                 'Import Directory': curdir,
                                 'Model directory': curdir,
                                 'Save Directory': curdir,
                                 'FOV': fovstr,
                                 'MaskAlpha': alpha,
                                 'SegThresh': .5,
                                 'PDWFthresh': thresh,
                                 'Preferred Import Type': 'dcm'}
            with open(self.configFN, 'w') as configfile:
                config.write(configfile)
                
        # parse config into self
        self.config = config
        self.fov = np.array([np.float(i) for i in self.config.get('DEFAULT','fov').split(",")])
        self.mask_alpha = np.float(self.config.get('DEFAULT','MaskAlpha'))
        self.segThresh = np.float(self.config.get('DEFAULT','SegThresh'))
        self.PDWFthresh = np.float(self.config.get('DEFAULT','PDWFthresh'))
        self.prefImpType = self.config.get('DEFAULT','Preferred Import Type')
        self.matdir = self.config.get('DEFAULT','mat directory')
        self.impdir = self.config.get('DEFAULT','import directory')
        self.mdldir = self.config.get('DEFAULT','model directory')
        self.savedir = self.config.get('DEFAULT','save directory')
##
    def setPreferences(self):
        # Open import window
        self.prefsW = PrefsWindow(self)
        self.prefsW.show()
        
    def Import(self):
        # check for saved
        if not self.saved:
            unsaved_msg = "You have unsaved data that will be discarded. Continue?"
            reply = QtWidgets.QMessageBox.question(self, 'Unsaved Data', 
                             unsaved_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        # clear current data
        self.savename = ''
        # Open import window
        self.ImpW = ImpWindow(self)
        self.ImpW.show()
        
##        
    def data_load(self,*args):
        # check for saved
        if not self.saved:
            unsaved_msg = "You have unsaved data that will be discarded. Continue?"
            reply = QtWidgets.QMessageBox.question(self, 'Unsaved Data', 
                             unsaved_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        # clear current data
        self.segmask = []
        self.segOutput = []
        self.curimagetype = 'water'
        self.bounddisp = False
        # Get filepath
        w = QtWidgets.QWidget()
        data_tup = QtWidgets.QFileDialog.getOpenFileName(w, 'Select data file',
                                                     self.savedir,
                                                       'MAT files (*.mat)',)
        data_path = data_tup[0]
        if len(data_path)==0:
            return
        self.disp_msg = "Loading previous data..."
        try:
            mat = spio.loadmat(data_path,squeeze_me=True)
            varbls = spio.whosmat(data_path)
            varbl_nms = [f[0] for f in varbls]
            if 'segfile' in varbl_nms:
                for key in varbl_nms:
                    try:
                        setattr(self, key, mat[key])
                        self.disp_msg = "Loaded " + key
                    except Exception as e:
                        print('Unable to load field',e)
                        pass
                self.savedir,self.savename = os.path.split(data_path)
                if len(self.imagesW)==0 and len(self.imagesF)!=0:
                    self.curimages = self.imagesF
                else:
                    self.curimages = self.imagesW
                self.disp_msg = "Previous data loaded"
                self.InitDisplay()
            else:
                self.error_msg = "Not a Seg file. Use import instead"
        except Exception as e:
            self.error_msg = "Unable to load data"
            print(e)
            
##
    def data_save(self,*args):
        if self.saved:
            self.disp_msg = 'No new changes'
            return
        try:
            w = QtWidgets.QWidget()
            if len(self.savename)==0:
                suggest = os.path.join(self.savedir,'SegData.mat')
            else:
                suggest = suggest = os.path.join(self.savedir,self.savename)
            save_path = QtWidgets.QFileDialog.getSaveFileName(w,'Save Data',suggest,"MAT file (*.mat)")
            
            if len(save_path[0])==0:
                return
            try:
                self.disp_msg = 'Saving data...'
                self.savedir,_ = os.path.split(save_path[0])
                a={}
                a['segfile'] = True
                a['imagesW'] = self.imagesW
                a['imagesF'] = self.imagesF
                a['inputs'] = self.inputs
                a['PDWFmap'] = self.PDWFmap
                a['segOutput'] = self.segOutput
                a['segmask'] = self.segmask
                a['uncor_segmask'] = self.uncor_segmask
                a['SIbounds'] = self.SIbounds
                a['vpoint'] = self.vpoint
                a['vpointL'] = self.vpointL
                a['vpointR'] = self.vpointR
                a['segThresh'] = self.segThresh
                a['spatres'] = self.spatres
                spio.savemat(save_path[0],a)
                self.saved = True
                self.disp_msg = 'Data saved'
            except Exception as e:
                self.error_msg = 'Error saving data'
                print(e)
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        except Exception as e:
            print(e)
##
    def load_model(self):
         # Get filepath
        w = QtWidgets.QWidget()
        data_tup = QtWidgets.QFileDialog.getOpenFileName(w, 'Select model file',
                                                         self.mdldir,
                                                         'HDF5 files (*.hdf5)',)
        data_path = data_tup[0]
        if len(data_path)==0:
            return
        self.disp_msg = "Loading model..."
        mdldir,_ = os.path.split(data_path)
        self.mdldir = mdldir
        self.ui.menubar.setEnabled(False)
        self.ui.progBar.setRange(0,0)
        self.ui.progBar.setVisible(True)
        QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        graph = keras.backend.tf.get_default_graph()
        self.load_thread = ModelLoadThread(data_path,graph)
        self.load_thread.finished.connect(self.loadModelFinish)
        self.load_thread.model_sig.connect(self.loadModelModel)
        self.load_thread.error_sig.connect(self.loadModelError)
        print('Starting model load thread...')
        self.load_thread.start()
        
    def load_model_weights(self):
        try:
            # Open model weights import dialog
            self.ModelDialog = LoadModelDialog(self)
            self.ModelDialog.show()
        except Exception as e:
            print(e)
            self.error_msg = 'Unable to load model from weights'
            
    def loadModelFinish(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.menubar.setEnabled(True)
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        
    def loadModelModel(self,model):
        self.disp_msg = 'Segmentation model loaded'
        self.model = model
        
    def loadModelError(self,msg):
        self.error_msg = msg
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        print(msg)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.menubar.setEnabled(True)
##
    def segment(self):
        if self.model==[]:
            self.error_msg = 'No model loaded'
            return
        try:
            self.prepInputs()
        except:
            self.error_msg = 'Unable to prepare inputs'
        try:
            # calculate batch sizes
            avail_mem = psutil.virtual_memory().available
            inpshape = self.inputs.shape
            needed_mem_per_batch = inpshape[1]*inpshape[2]*inpshape[3]*885
            batch_size_allowed = np.floor(.5*avail_mem/needed_mem_per_batch)
            num_batch = np.ceil(inpshape[0]/batch_size_allowed)
            import keras    
            self.graph = keras.backend.tf.get_default_graph()
                
            self.disp_msg = 'Starting segmentation...'
            self.ui.menubar.setEnabled(False)
            self.ui.progBar.setRange(0,0)
            self.ui.progBar.setVisible(True)
            QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            self.seg_thread = SegThread(self.graph,self.inputs,self.model,num_batch)
            self.seg_thread.error_sig.connect(self.loadModelError)
            self.seg_thread.finished.connect(self.segFinish)
            self.seg_thread.batch_sig.connect(self.segBatch)
            self.seg_thread.segmask_sig.connect(self.segGotMask)
            
            self.seg_thread.start()
        except Exception as e:
            print(e)
            return
    
    def prepInputs(self):
        self.disp_msg = 'Preparing inputs for segmentation...'
        if len(self.imagesW)==0 or len(self.imagesF)==0:
            self.error_msg = "No data loaded"
            raise
        try:
            # normalize
            imagesW = np.float32(self.imagesW)
            imagesF = np.float32(self.imagesF)
            for ss in range(imagesW.shape[0]):
                np.divide(imagesW[ss,:,:],np.max(imagesW[ss,:,:]),out=imagesW[ss,:,:])
                np.divide(imagesF[ss,:,:],np.max(imagesF[ss,:,:]),out=imagesF[ss,:,:])
            if self.model.get_input_shape_at(0)[3]==3:
                # get Vcut
                Vcut = self.makeVcut()
                Vmask = np.tile(Vcut[np.newaxis,...],(self.volshape[0],1,1))
                # concatenate channels
                self.inputs = np.stack((imagesW,imagesF,Vmask),axis=3)
            else:
                self.inputs = np.stack((imagesW,imagesF),axis=3)
                
            self.disp_msg = "Input image preparation complete"
        except Exception as e:
            print(e)
            raise
            
    def makeVcut(self):
        imshape = (self.volshape[1],self.volshape[2])
        r = np.array([self.vpointL[1],self.vpoint[1],self.vpointR[1],
                      self.volshape[1],self.volshape[1],self.vpointL[1]])
        c = np.array([self.vpointL[0],self.vpoint[0],self.vpointR[0],
                      self.volshape[2],0,self.vpointL[0]])
        rr, cc = polygon(r, c,shape=imshape)
        Vcut = np.zeros(imshape,dtype=np.float)
        Vcut[rr,cc] = 1
        return Vcut
            
    def segFinish(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.menubar.setEnabled(True)
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        try:
            self.maskdisp = True
            self.maskOn()
            self.resetView()
            self.disp_msg = 'Segmentation complete'
        except Exception as e:
            print(e)
    def segBatch(self,addNum):
        self.ui.progBar.setRange(0,self.inputs.shape[0])
        self.ui.progBar.setValue(self.ui.progBar.value()+addNum)
    def segGotMask(self,output):
        self.segOutput = output
        self.segmask = np.clip(np.float32(self.segOutput>self.segThresh),0,1)
        time.sleep(.01)
        self.uncor_segmask = np.copy(self.segmask)
##
    def setSegThresh(self):
        if len(self.segOutput)==0:
            self.error_msg = 'No segmentation data'
            return
        self.slideDialog = ThresholdSlider(self)
        self.slideDialog.show()
        
    def updateMaskThresh(self):
        try:
            # remake mask
            self.segmask = np.float32(self.segOutput>self.segThresh)
            self.uncor_segmask = np.copy(self.segmask)
            self.mask[...,3] = self.mask_alpha*self.segmask
            # update display
            self.msk_item_ax.setImage(self.mask[self.inds[0],...])
            self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])
            self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
        except Exception as e:
            print(e)
##
    def openCorrector(self):
        # Open corrections window
        if len(self.imagesW)==0 and len(self.imagesW)==0:
            self.error_msg = 'No images loaded'
            return
        
        levs = self.img_item_ax.getLevels()
        self.CorW = CorWindow(self,levs,self.inds[0])
        self.CorW.show()

    def InitDisplay(self):
        try:
            # calculate aspect ratios
            asprat1 = self.spatres[1]/self.spatres[2]
            asprat2 = self.spatres[1]/self.spatres[0]
            asprat3 = self.spatres[2]/self.spatres[0]
            ## add View Box to graphics view
            if not hasattr(self, 'vbox_ax'):
                self.vbox_ax = self.ui.viewAxial.addViewBox(border=None,
                                                            enableMenu=False,
                                                            enableMouse=False,
                                                            invertY=True)
                self.vbox_cor = self.ui.viewCoronal.addViewBox(border=None,
                                                               enableMenu=False,
                                                               enableMouse=False)
                self.vbox_sag = self.ui.viewSag.addViewBox(border=None,
                                                           enableMenu=False,
                                                           enableMouse=False)
            else:
                self.vbox_ax.clear()
                self.vbox_cor.clear()
                self.vbox_sag.clear()
                
            self.vbox_ax.setAspectLocked(True,ratio=asprat1)
            self.vbox_cor.setAspectLocked(True,ratio=asprat2)
            self.vbox_sag.setAspectLocked(True,ratio=asprat3)
                
            ## Create image items
            self.img_item_ax = pg.ImageItem()
            self.vbox_ax.addItem(self.img_item_ax)
            
            self.img_item_cor = pg.ImageItem()
            self.vbox_cor.addItem(self.img_item_cor)
            
            self.img_item_sag = pg.ImageItem()
            self.vbox_sag.addItem(self.img_item_sag)
            
            # Create mask items
            self.msk_item_ax = pg.ImageItem()
            self.vbox_ax.addItem(self.msk_item_ax)
            
            self.msk_item_cor = pg.ImageItem()
            self.vbox_cor.addItem(self.msk_item_cor)
            
            self.msk_item_sag = pg.ImageItem()
            self.vbox_sag.addItem(self.msk_item_sag)    
            
            # Setup indices
            imshape = np.array(self.imagesW.shape,dtype=np.int16)
            midinds = np.round(imshape/2).astype(np.int16)
            self.inds = midinds
            self.volshape = imshape
            
            # Add initial images            
            self.img_item_ax.setImage(self.curimages[midinds[0],...])
            self.img_item_cor.setImage(self.curimages[:,midinds[1],:])
            self.img_item_sag.setImage(self.curimages[:,:,midinds[2]])
            
            # add masks, even if empty
            msksiz = np.r_[self.imagesW.shape,4]
            msk = np.zeros(msksiz,dtype=np.float)
            msk[...,0] = 1
            msk[...,1] = .4
            msk[...,2] = .4
            if not len(self.segmask)==0:
                msk[...,3] = self.mask_alpha*self.segmask.astype(np.float)
                self.maskdisp = True
            self.mask = msk
            self.msk_item_ax.setImage(self.mask[midinds[0],...])
            self.msk_item_cor.setImage(self.mask[:,midinds[1],...])
            self.msk_item_sag.setImage(self.mask[:,:,midinds[2],...])
            
            # Create plot items
            # S/I boundaries
            SIpen = pg.mkPen(color=(0, 50, 255, 120),width=2)
            if len(self.SIbounds)==0:
                SIbounds = np.array([1,imshape[0]-1],dtype=np.int16)
                self.SIbounds = SIbounds
            else:
                SIbounds = self.SIbounds
                
            # coronal
            self.line_cor_S = pg.PlotDataItem(pen=SIpen,connect='all')
            self.line_cor_S.setData(x = np.array([0,1]),
                                     y = np.array([SIbounds[1],SIbounds[1]]))
            self.vbox_cor.addItem(self.line_cor_S)
            self.line_cor_I = pg.PlotDataItem(pen=SIpen,connect='all')
            self.line_cor_I.setData(x = np.array([0,1]),
                                     y = np.array([SIbounds[0],SIbounds[0]]))
            self.vbox_cor.addItem(self.line_cor_I)
            # sagittal
            self.line_sag_S = pg.PlotDataItem(pen=SIpen,connect='all')
            self.line_sag_S.setData(x = np.array([0,1]),
                                     y = np.array([SIbounds[1],SIbounds[1]]))
            self.vbox_sag.addItem(self.line_sag_S)
            self.line_sag_I = pg.PlotDataItem(pen=SIpen,connect='all')
            self.line_sag_I.setData(x = np.array([0,1]),
                                     y = np.array([SIbounds[0],SIbounds[0]]))
            self.vbox_sag.addItem(self.line_sag_I)
            
            # V cut boundaries
            Vbrush = pg.mkBrush(color=(0, 50, 255, 120))
            # estimate initial location
            # vertex point
            if len(self.vpoint)==0:
                Vpoint = np.round(np.array([imshape[2]/2,.92*imshape[1]]))
                self.vpoint = Vpoint
            else:
                Vpoint = self.vpoint
            self.vpoint_plot = pg.ScatterPlotItem(pxMode=False,size=6,symbol='+',brush=Vbrush,pen=SIpen)
            self.vpoint_plot.setData(x=np.array([Vpoint[0]]),
                                     y=np.array([Vpoint[1]]))
            self.vbox_ax.addItem(self.vpoint_plot)
            # right side point
            delY = np.round(Vpoint[0]*np.tan(40/180*np.pi)) # fixed 100 degree angle=40 degrees complement on each side
            VY = Vpoint[1]-delY
            VpointR = np.array([imshape[2],VY])
            self.vpointR = VpointR
            self.vline_r = pg.PlotCurveItem(pen=SIpen,connect='all')
            self.vline_r.setData(x = np.array([Vpoint[0],VpointR[0]]),
                                 y = np.array([Vpoint[1],VpointR[1]]))
            self.vline_r.setClickable(True, width=10)
            self.vbox_ax.addItem(self.vline_r)
            # left side point
            self.vpointL = np.array([0,VY])
            self.vline_l = pg.PlotCurveItem(pen=SIpen,connect='all')
            self.vline_l.setData(x = np.array([Vpoint[0],0]),
                                 y = np.array([Vpoint[1],VpointR[1]]))
            self.vline_l.setClickable(True, width=10)
            self.vbox_ax.addItem(self.vline_l)
            self.boundsOff()
            
            #calculate buffer
            buff = 13 #in mm
            self.buff = np.round(np.array(buff/self.spatres)).astype(np.int16)
            # coronal view lines
            self.line_cor_ax = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_cor_ax.setData(x = np.array([0,midinds[2]-self.buff[2],midinds[2]+self.buff[2],imshape[2]]),
                                     y = np.array([midinds[0],midinds[0],midinds[0],midinds[0]]))
            self.vbox_cor.addItem(self.line_cor_ax)
            
            self.line_cor_sag = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_cor_sag.setData(x = np.array([midinds[1],midinds[1],midinds[1],midinds[1]]),
                                     y = np.array([0,midinds[0]-self.buff[0],midinds[0]+self.buff[0],imshape[0]]))
            self.vbox_cor.addItem(self.line_cor_sag)
            # saggital view lines
            self.line_sag_ax = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_sag_ax.setData(x = np.array([0,midinds[1]-self.buff[1],midinds[1]+self.buff[1],imshape[1]]),
                                     y = np.array([midinds[0],midinds[0],midinds[0],midinds[0]]))
            self.vbox_sag.addItem(self.line_sag_ax)
            
            self.line_sag_cor = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_sag_cor.setData(x = np.array([midinds[2],midinds[2],midinds[2],midinds[2]]),
                                     y = np.array([0,midinds[0]-self.buff[0],midinds[0]+self.buff[0],imshape[0]]))
            self.vbox_sag.addItem(self.line_sag_cor)
            
            # axial view lines
            self.line_ax_cor = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_ax_cor.setData(x = np.array([0,midinds[2]-self.buff[2],midinds[2]+self.buff[2],imshape[2]]),
                                     y = np.array([midinds[1],midinds[1],midinds[1],midinds[1]]))
            self.vbox_ax.addItem(self.line_ax_cor)
            
            self.line_ax_sag = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_ax_sag.setData(x = np.array([midinds[2],midinds[2],midinds[2],midinds[2]]),
                                     y = np.array([0,midinds[1]-self.buff[1],midinds[1]+self.buff[1],imshape[2]]))
            self.vbox_ax.addItem(self.line_ax_sag)
            
            # adjust viewbox ranges
            self.vbox_ax.setRange(xRange=(0,imshape[2]),yRange=(0,imshape[1]),
                                  disableAutoRange=True,padding=0.)
            self.vbox_cor.setRange(xRange=(0,imshape[2]),yRange=(0,imshape[0]),
                                  disableAutoRange=True,padding=0.)
            self.vbox_sag.setRange(xRange=(0,imshape[1]),yRange=(0,imshape[0]),
                                  disableAutoRange=True,padding=0.)
            # calculate window/leveling multiplier
            self.WLmult = self.img_item_ax.getLevels()[1]/500
            
            # Setup slider
            self.ui.slideAx.setMaximum(self.imagesW.shape[0]-1)
            self.ui.slideAx.setValue(midinds[0])
                        
            # attach callbacks
            self.ui.slideAx.valueChanged.connect(self.axSlide)
            
            self.ui.viewAxial.wheelEvent = self.axScroll
            self.img_item_ax.mouseClickEvent = self.axClickEvent
            self.img_item_ax.mouseDragEvent = self.axDragEvent
            
            self.ui.viewCoronal.wheelEvent = self.corScroll
            self.img_item_cor.mouseClickEvent = self.corClickEvent
            self.img_item_cor.mouseDragEvent = self.corDragEvent
            
            self.ui.viewSag.wheelEvent = self.sagScroll
            self.img_item_sag.mouseClickEvent = self.sagClickEvent
            self.img_item_sag.mouseDragEvent = self.sagDragEvent
            
            self.ui.viewAxial.scene().setMoveDistance(1)
            self.ui.viewCoronal.scene().setMoveDistance(1)
            self.ui.viewSag.scene().setMoveDistance(1)
            # attach context menus
            self.ui.viewAxial.customContextMenuRequested.connect(self.axCMenu)
            
            # make full screen
#            self.showMaximized()
            scrnsiz = QtWidgets.QDesktopWidget().screenGeometry()
            cntr = scrnsiz.center()
            width = 1011
            height = 675
            xv = np.round(cntr.x()-width/2)
            yv = np.round(cntr.y()-height/2)
            sizRect = QtCore.QRect(xv,yv,width,height)
            self.setGeometry(sizRect)
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        
    def updateIms(self):
        self.img_item_ax.setImage(self.curimages[self.inds[0],...],autoLevels=False)
        self.img_item_cor.setImage(self.curimages[:,self.inds[1],:],autoLevels=False)
        self.img_item_sag.setImage(self.curimages[:,:,self.inds[2]],autoLevels=False)
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
        
    def updateLines(self):
        self.line_ax_cor.setData(x = np.array([0,self.inds[2]-self.buff[2],self.inds[2]+self.buff[2],self.volshape[2]]),
                                 y = np.array([self.inds[1],self.inds[1],self.inds[1],self.inds[1]]))
        self.line_ax_sag.setData(x = np.array([self.inds[2],self.inds[2],self.inds[2],self.inds[2]]),
                                 y = np.array([0,self.inds[1]-self.buff[1],self.inds[1]+self.buff[1],self.volshape[1]]))
        
        self.line_cor_ax.setData(x = np.array([0,self.inds[2]-self.buff[2],self.inds[2]+self.buff[2],self.volshape[2]]),
                                 y = np.array([self.inds[0],self.inds[0],self.inds[0],self.inds[0]]))
        self.line_cor_sag.setData(x = np.array([self.inds[2],self.inds[2],self.inds[2],self.inds[2]]),
                                  y = np.array([0,self.inds[0]-self.buff[0],self.inds[0]+self.buff[0],self.volshape[0]]))
        
        self.line_sag_ax.setData(x = np.array([0,self.inds[1]-self.buff[1],self.inds[1]+self.buff[1],self.volshape[1]]),
                                 y=np.array([self.inds[0],self.inds[0],self.inds[0],self.inds[0]]))
        self.line_sag_cor.setData(x = np.array([self.inds[1],self.inds[1],self.inds[1],self.inds[1]]),
                             y = np.array([0,self.inds[0]-self.buff[0],self.inds[0]+self.buff[0],self.volshape[0]]))
        
        
    def axSlide(self,event):
        self.img_item_ax.setImage(self.curimages[event,...],autoLevels=False)
        self.msk_item_ax.setImage(self.mask[event,...])
        self.inds[0] = event
        self.updateLines()
        
    def axScroll(self,event):
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods == QtCore.Qt.ControlModifier:
            bump = -event.angleDelta().y()/120
            fac = 1+.1*bump
            pos = self.vbox_ax.mapToView(event.pos())
            self.vbox_ax.scaleBy(s=fac,center=(pos.x(),pos.y()))
            self.vbox_cor.scaleBy(s=fac,center=(pos.x(),self.inds[0]))
            self.vbox_sag.scaleBy(s=fac,center=(pos.y(),self.inds[0]))
            event.accept()
        else:
            curval = self.ui.slideAx.value()
            newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                             self.volshape[0]-1))
            self.ui.slideAx.setValue(newval)
        
    def corScroll(self,event):
        curval = self.inds[1]
        newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                         self.volshape[1]-1))
        self.inds[1]= newval
        self.corUpdate(newval)
        self.updateLines()
                 
    def corUpdate(self,value):
        self.img_item_cor.setImage(self.curimages[:,value,:],autoLevels=False)
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])       
         
    def sagScroll(self,event):
        curval = self.inds[2]
        newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                         self.volshape[2]-1))
        self.inds[2] = newval
        self.sagUpdate(newval)
        self.updateLines()
            
    def sagUpdate(self,value):
        self.img_item_sag.setImage(self.curimages[:,:,value],autoLevels=False)
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
        
    def axClickEvent(self,ev):
        if ev.button()==1:
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.axMove(posx,posy)
            ev.accept()
        else:
            ev.ignore()
                    
    def axDragEvent(self,ev):
        try:
            if ev.button()==1:
                if ev.isStart():
                    self.rbound = self.vline_r.mouseShape()
                    self.lbound = self.vline_l.mouseShape()
                    self.vbound = self.vpoint_plot.boundingRect()
                point = ev.buttonDownPos()
                if self.bounddisp and self.vbound.contains(point):
                    self.axVpointMove(ev)
                elif self.bounddisp and self.rbound.contains(point):
                    self.axVlineRmove(ev)
                elif self.bounddisp and self.lbound.contains(point):
                    self.axVlineLmove(ev)
                else:
                    posx = ev.pos().x()
                    posy = ev.pos().y()
                    self.axMove(posx,posy)
                ev.accept()
                
            if ev.button()==4:
                    if ev.isStart():
                        self.prevLevel = np.array(self.img_item_ax.getLevels())
                        self.startPos = np.array([ev.pos().x(),ev.pos().y()])
                    else:
                        self.LevelEvent(ev)
                    ev.accept()
        except Exception as e:
            print(e)
    def axVpointMove(self,ev):
        oldRX,oldRY = self.vline_r.getData()
        oldLX,oldLY = self.vline_l.getData()
        newX = np.clip(np.round(ev.pos().x()),0,self.volshape[2])
        newY = np.clip(np.round(ev.pos().y()),0,self.volshape[1])
        self.vpoint_plot.setData(x = np.array([newX]),
                                 y = np.array([newY]))
        self.vline_r.setData(x = np.array([newX,oldRX[1]]),
                             y = np.array([newY,oldRY[1]]))
        self.vline_l.setData(x = np.array([newX,oldLX[1]]),
                             y = np.array([newY,oldLY[1]]))
        if ev.isFinish():
            self.vpoint = np.array([newX,newY])
            self.saved = False
            
    def axVlineRmove(self,ev):
        curX = np.round(np.clip(ev.pos().x(),self.volshape[2]/2+1,self.volshape[2]))
        curY = np.round(np.clip(ev.pos().y(),0,self.volshape[1]))
        vX,vY = self.vpoint
        m = (curY-vY)/(curX-vX)
        newX = self.volshape[2]
        newY = np.round(np.minimum(m*(newX-vX) + vY,self.volshape[1]))
        self.vline_r.setData(x = np.array([vX,newX]),
                             y = np.array([vY,newY]))
        if ev.isFinish():
            self.vpointR = np.array([newX,newY])
            self.saved = False
    def axVlineLmove(self,ev):
        curX = np.round(np.clip(ev.pos().x(),0,self.volshape[2]/2-1))
        curY = np.round(np.clip(ev.pos().y(),0,self.volshape[1]))
        vX,vY = self.vpoint
        m = (curY-vY)/(curX-vX)
        newX = 0
        newY = np.round(np.minimum(m*(newX-vX) + vY,self.volshape[1]))
        self.vline_l.setData(x = np.array([vX,newX]),
                             y = np.array([vY,newY]))
        if ev.isFinish():
            self.vpointL = np.array([newX,newY])
            self.saved = False
    
    def axMove(self,x,y):
        cval = np.int16(np.clip(y,0,self.volshape[1]-1))
        self.inds[1]= cval
        self.corUpdate(cval)
        sval = np.int16(np.clip(x,0,self.volshape[2]-1))
        self.inds[2]= sval
        self.sagUpdate(sval)
        self.updateLines()
        
    def corClickEvent(self,ev):
        if ev.button()==1:
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.corMove(posx,posy)
            ev.accept()
        else:
            ev.ignore()
                    
    def corDragEvent(self,ev):
        if ev.button()==1:
            clickPos = ev.buttonDownPos()
            if np.any(np.abs(clickPos.y()-self.SIbounds)<2) and self.bounddisp:
                self.corLineDrag(ev)
            else:
                posx = ev.pos().x()
                posy = ev.pos().y()
                self.corMove(posx,posy)
            ev.accept()
            
        if ev.button()==4:
                if ev.isStart():
                    self.prevLevel = np.array(self.img_item_cor.getLevels())
                    self.startPos = np.array([ev.pos().x(),ev.pos().y()])
                else:
                    self.LevelEvent(ev)
                ev.accept()
                
    def corMove(self,x,y):
        aval = np.int16(np.clip(y,0,self.volshape[0]-1))
        self.inds[0] = aval
        self.ui.slideAx.setValue(aval)
        sval = np.int16(np.clip(x,0,self.volshape[2]-1))
        self.inds[2] = sval
        self.sagUpdate(sval)
        self.updateLines()
        
    def corLineDrag(self,ev):
        clickY = ev.buttonDownPos().y()
        newY = np.clip(np.round(ev.pos().y()).astype(np.int),0,self.volshape[0]-1)
        dist = np.abs(clickY-self.SIbounds)
        if dist[0]>dist[1]:
            self.line_cor_S.setData(x = np.array([0,self.volshape[2]]),
                                     y = np.array([newY,newY]))
            self.line_sag_S.setData(x = np.array([0,self.volshape[1]]),
                                     y = np.array([newY,newY]))
            if ev.isFinish():
                self.SIbounds[1] = newY
                self.ui.slideAx.setValue(newY)
                self.saved = False
        else:
            self.line_cor_I.setData(x = np.array([0,self.volshape[2]]),
                                     y = np.array([newY,newY]))
            self.line_sag_I.setData(x = np.array([0,self.volshape[1]]),
                                     y = np.array([newY,newY]))
            if ev.isFinish():
                self.SIbounds[0] = newY
                self.ui.slideAx.setValue(newY)
                self.saved = False
        self.img_item_ax.setImage(self.curimages[newY,...],autoLevels=False)
        self.msk_item_ax.setImage(self.mask[newY,...])
                                 
    def sagClickEvent(self,ev):
        if ev.button()==1:
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.sagMove(posx,posy)
            ev.accept()
        else:
            ev.ignore()
            
    def sagDragEvent(self,ev):
        if ev.button()==1:
            clickPos = ev.buttonDownPos()
            if np.any(np.abs(clickPos.y()-self.SIbounds)<2) and self.bounddisp:
                self.sagLineDrag(ev)
            else:
                posx = ev.pos().x()
                posy = ev.pos().y()
                self.sagMove(posx,posy)
            ev.accept()
            
        if ev.button()==4:
                if ev.isStart():
                    self.prevLevel = np.array(self.img_item_sag.getLevels())
                    self.startPos = np.array([ev.pos().x(),ev.pos().y()])
                else:
                    self.LevelEvent(ev)
                ev.accept()
                
    def sagMove(self,x,y):
        aval = np.int16(np.clip(y,0,self.volshape[0]-1))
        self.inds[0] = aval
        self.ui.slideAx.setValue(aval)
        cval = np.int16(np.clip(x,0,self.volshape[1]-1))
        self.inds[1] = cval
        self.corUpdate(cval)
        self.updateLines()
    
    def sagLineDrag(self,ev):
        clickY = ev.buttonDownPos().y()
        newY = np.clip(np.round(ev.pos().y()).astype(np.int),0,self.volshape[0]-1)
        dist = np.abs(clickY-self.SIbounds)
        if dist[0]>dist[1]:
            self.line_cor_S.setData(x = np.array([0,self.volshape[2]]),
                                     y = np.array([newY,newY]))
            self.line_sag_S.setData(x = np.array([0,self.volshape[1]]),
                                     y = np.array([newY,newY]))
            if ev.isFinish():
                    self.SIbounds[1] = newY
                    self.ui.slideAx.setValue(newY)
                    self.saved = False
        else:
            self.line_cor_I.setData(x = np.array([0,self.volshape[2]]),
                                     y = np.array([newY,newY]))
            self.line_sag_I.setData(x = np.array([0,self.volshape[1]]),
                                     y = np.array([newY,newY]))
            if ev.isFinish():
                    self.SIbounds[0] = newY
                    self.ui.slideAx.setValue(newY)
                    self.saved = False
        self.img_item_ax.setImage(self.curimages[newY,...],autoLevels=False)
        self.msk_item_ax.setImage(self.mask[newY,...])
        
    def LevelEvent(self,ev):
        curpos = np.array([ev.pos().x(),ev.pos().y()])
        posdiff = self.WLmult*(curpos-self.startPos)
        prevLevels = self.prevLevel
        newW = prevLevels[1]-prevLevels[0]+posdiff[0]
        newL = np.mean(prevLevels)+posdiff[1]
        newLev0 = newL-newW/2
        newLev1 = newL+newW/2
        self.img_item_ax.setLevels([newLev0,newLev1])
        self.img_item_cor.setLevels([newLev0,newLev1])
        self.img_item_sag.setLevels([newLev0,newLev1])
        ev.accept()
    
    def axCMenu(self,position):
        menu = QtWidgets.QMenu()
        watimAction = QtWidgets.QAction("Water Images",menu,checkable=True)
        fatimAction = QtWidgets.QAction("Fat Images",menu,checkable=True)
        if self.curimagetype=='water':
            watimAction.setChecked(True)
        elif self.curimagetype=='fat':
            fatimAction.setChecked(True)
        if len(self.imagesW)!=0:
            menu.addAction(watimAction)
        if len(self.imagesF)!=0:
            menu.addAction(fatimAction)
        menu.addSeparator()
        togmaskAction = QtWidgets.QAction("Display mask", menu, checkable=True)
        togmaskAction.setChecked(self.maskdisp)
        menu.addAction(togmaskAction)
        togboundsAction = QtWidgets.QAction("Display bounds", menu, checkable=True)
        togboundsAction.setChecked(self.bounddisp)
        menu.addAction(togboundsAction)
        segthreshAction = menu.addAction("Adjust Segmentation Threshold...")
        testAction = menu.addAction("Test Action")
        menu.addSeparator()
        resetAction = menu.addAction("Reset View")
        action = menu.exec_(self.ui.viewAxial.mapToGlobal(position))
        if action == fatimAction:
            self.curimages = self.imagesF
            self.curimagetype = 'fat'
            self.updateIms()
        elif action == watimAction:
            self.curimages = self.imagesW
            self.curimagetype = 'water'
            self.updateIms()
        elif action == togmaskAction:
            try:
                if not len(self.segmask)==0:
                    if self.maskdisp:
                        self.maskOff()
                        self.maskdisp = False
                    else:
                        self.maskOn()
                        self.maskdisp = True
                else:
                    self.error_msg = 'No seg mask'
            except Exception as e:
                print(e)
        elif action == togboundsAction:
            if self.bounddisp:
                self.bounddisp = False
                self.boundsOff()
            else:
                self.bounddisp = True
                self.boundsOn()
        elif action == segthreshAction:
            self.setSegThresh()
        elif action == testAction:
            print('Test action')
        elif action == resetAction:
            self.resetView()
        menu.close()
    def maskOn(self):
        # show mask
        try:
            self.mask[...,3] = .3*self.segmask.astype(np.float)
        except Exception as e:
            print(e)
            pass
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
    def maskOff(self):
        # turn mask off
        self.mask[...,3] = 0
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
        
    def boundsOn(self):
        try:
            self.line_cor_S.setData(x = np.array([0,self.volshape[2]]),
                                    y = np.array([self.SIbounds[1],self.SIbounds[1]]))
            self.line_cor_I.setData(x = np.array([0,self.volshape[2]]),
                                    y = np.array([self.SIbounds[0],self.SIbounds[0]]))
            self.line_sag_S.setData(x = np.array([0,self.volshape[1]]),
                                    y = np.array([self.SIbounds[1],self.SIbounds[1]]))
            self.line_sag_I.setData(x = np.array([0,self.volshape[1]]),
                                    y = np.array([self.SIbounds[0],self.SIbounds[0]]))
                    
            self.vpoint_plot.setData(x = np.array([self.vpoint[0]]),
                                     y = np.array([self.vpoint[1]]))
            self.vline_r.setData(x = np.array([self.vpoint[0],self.vpointR[0]]),
                                     y = np.array([self.vpoint[1],self.vpointR[1]]))
            self.vline_l.setData(x = np.array([self.vpoint[0],self.vpointL[0]]),
                                     y = np.array([self.vpoint[1],self.vpointL[1]]))
        except Exception as e:
            print(e)
    def boundsOff(self):
        try:
            self.line_cor_S.setData(x = np.array([0,1]),
                                    y = np.array([self.SIbounds[1],self.SIbounds[1]]))
            self.line_cor_I.setData(x = np.array([0,1]),
                                    y = np.array([self.SIbounds[0],self.SIbounds[0]]))
            self.line_sag_S.setData(x = np.array([0,1]),
                                    y = np.array([self.SIbounds[1],self.SIbounds[1]]))
            self.line_sag_I.setData(x = np.array([0,1]),
                                    y = np.array([self.SIbounds[0],self.SIbounds[0]]))
            self.vpoint_plot.setData(x = np.array([-20]),
                                     y = np.array([self.volshape[1]+20]))
            self.vline_l.setData(x = np.array([-10]),
                                 y = np.array([self.volshape[1]+10]))
            self.vline_r.setData(x = np.array([-10]),
                                 y = np.array([self.volshape[1]+10]))
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
    def resetView(self):
        # reset viewbox ranges
        self.vbox_ax.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[1]))
        self.vbox_cor.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[0]))
        self.vbox_sag.setRange(xRange=(0,self.volshape[1]),yRange=(0,self.volshape[0]))
        
        # reset drawings
        self.inds = np.round(self.volshape/2).astype('int16')
        self.img_item_ax.setImage(self.curimages[self.inds[0],...],autoLevels=True)
        self.img_item_cor.setImage(self.curimages[:,self.inds[1],:],autoLevels=True)
        self.img_item_sag.setImage(self.curimages[:,:,self.inds[2]],autoLevels=True)
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])
        self.updateLines()
        
    def shiftDims(self):
        try:
            self.imagesW = np.rollaxis(self.imagesW,2,0)
            self.imagesF = np.rollaxis(self.imagesF,2,0)
            self.volshape = np.roll(self.volshape,1)
            self.inds = np.roll(self.inds,1)
            self.PDWFmap = np.rollaxis(self.PDWFmap,2,0)
            self.mask = np.rollaxis(self.mask,2,0)
            self.segOutput = np.rollaxis(self.segOutput,2,0)
            self.segmask = np.rollaxis(self.segmask,2,0)
            self.uncor_segmask = np.rollaxis(self.uncor_segmask,2,0)
        except Exception as e:
            print(e)
        try:
            # calculate aspect ratios
            self.spatres = np.roll(self.spatres,1)
            asprat1 = self.spatres[1]/self.spatres[2]
            asprat2 = self.spatres[1]/self.spatres[0]
            asprat3 = self.spatres[2]/self.spatres[0]
            self.vbox_ax.setAspectLocked(True,ratio=asprat1)
            self.vbox_cor.setAspectLocked(True,ratio=asprat2)
            self.vbox_sag.setAspectLocked(True,ratio=asprat3)
            self.ui.slideAx.setMaximum(self.volshape[0]-1)
            self.ui.slideAx.setValue(self.inds[0])
            if self.curimagetype=='water':
                self.curimages = self.imagesW
            elif self.curimagetype=='fat':
                self.curimages = self.imagesF
        except Exception as e:
            print(e)
        self.updateIms()
        self.updateLines()
        
    def flipIms(self,axis):
        try:
            self.imagesW = np.flip(self.imagesW,axis)
            self.imagesF = np.flip(self.imagesF,axis)
            if len(self.PDWFmap)==0:
                self.PDWFmap = np.flip(self.PDWFmap,axis)
            if len(self.mask)==0:
                self.mask = np.flip(self.mask,axis)
            if len(self.segOutput) == 0:
                self.segOutput = np.flip(self.segOutput,axis)
            if len(self.segmask) == 0:
                self.segmask = np.flip(self.segmask,axis)
            if len(self.uncor_segmask) == 0:
                self.uncor_segmask = np.flip(self.uncor_segmask,axis)
        except Exception as e:
            print(e)
        if self.curimagetype=='water':
            self.curimages = self.imagesW
        elif self.curimagetype=='fat':
            self.curimages = self.imagesF
        self.updateIms()
        self.updateLines()
        

    def calcPDWF(self):
        try:
            self.disp_msg = 'Calculating PDWF value...'
            if len(self.PDWFmap)==0:
                self.error_msg = "No PDWF map"
                return
            if  len(self.segmask)==0:
                self.error_msg = 'No Segmentation Mask'
                return
            PDWFmask = np.multiply(self.PDWFmap,self.segmask)
            PDWFmask[PDWFmask>self.PDWFthresh] = self.PDWFthresh
            PDWFmask[PDWFmask<-self.PDWFthresh] = -self.PDWFthresh
            voxvol = np.prod(self.spatres)
            water_vol = np.abs(np.sum(PDWFmask))*voxvol
            tot_vol = np.sum(self.segmask.astype(np.float))*voxvol
            self.PDWF = water_vol/tot_vol
            self.disp_msg = "Water volume is {:.3f} cc".format(water_vol/1000)
            self.disp_msg = "Total volume is {:.3f} cc".format(tot_vol/1000)
            self.disp_msg = "PDWF is {:.2%}".format(self.PDWF)
        except Exception as e:
            print(e)
    
    def resetMask(self):
        check_msg = "Are you sure you wish to discard all corrections?"
        reply = QtWidgets.QMessageBox.question(self, 'Reset Mask', 
                         check_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.No:
            return
        # remake mask
        if not len(self.segOutput)==0:
            self.segmask = np.float32(self.segOutput>self.segThresh)
            self.uncor_segmask = np.copy(self.segmask)
            self.mask[...,3] = self.mask_alpha*self.segmask
        else:
            self.segmask = np.copy(self.uncor_segmask)
            self.mask[...,3] = self.mask_alpha*self.segmask
        
        self.updateIms()
        
    
    def openHelp(self):
        file = 'UserManual.pdf'
        try:
            os.startfile(file)
        except Exception as e:
            self.error_msg = 'Unable to open help file'
        
    @pyqtProperty(str)
    def disp_msg(self):
            return self.disp_msg
    @disp_msg.setter
    def disp_msg(self, value):
        self._disp_msg = value
        item = QtWidgets.QListWidgetItem(value)
        self.ui.listDisp.addItem(item)
        if self.ui.listDisp.count() > 200:
            self.ui.listDisp.takeItem(0)
        self.ui.listDisp.scrollToBottom()
        self.ui.listDisp.update()
        print(value)
        QtWidgets.qApp.processEvents()
        
    @pyqtProperty(str)
    def error_msg(self):
            return self._error_msg
    @error_msg.setter
    def error_msg(self, value):
        self._error_msg = value
        item = QtWidgets.QListWidgetItem(value)
        item.setForeground(QtCore.Qt.red)
        self.ui.listDisp.addItem(item)
        if self.ui.listDisp.count() > 200:
            self.ui.listDisp.takeItem(0)
        self.ui.listDisp.scrollToBottom()
        self.ui.listDisp.update()
        print('Error:',value)
        QtWidgets.qApp.processEvents()
    def saveConfig(self):
        self.config['DEFAULT']['mat directory'] = self.matdir
        self.config['DEFAULT']['import directory'] = self.impdir
        self.config['DEFAULT']['model directory'] = self.mdldir
        self.config['DEFAULT']['save directory'] = self.savedir
        self.config['DEFAULT']['fov'] = ','.join(str(x) for x in self.fov)
        self.config['DEFAULT']['SegThresh'] = str(self.segThresh)
        self.config['DEFAULT']['MaskAlpha'] = str(self.mask_alpha)
        self.config['DEFAULT']['PDWFthresh'] = str(self.PDWFthresh)
        self.config['DEFAULT']['Preferred Import Type'] = self.prefImpType
        with open(self.configFN, 'w') as configfile:
            self.config.write(configfile)
        
    def closeEvent(self, ev):
        # check for saved
        if not self.saved:
            quit_msg = "You have unsaved data. Are you sure you wish to close?"
            reply = QtWidgets.QMessageBox.warning(self, 'Closing', 
                             quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        
            if reply == QtWidgets.QMessageBox.Yes:
                ev.accept()
            else:
                ev.ignore()
                return
            
        # save config file
        self.saveConfig()
        # exit
        try:
            for child in [self.CorW]:
                child.destroy()
        except Exception as e:
            pass
        self.destroy()

#%%
class DataSelect(QtBaseClass2,Ui_DataSelect):
    def __init__(self,parent):
        super(DataSelect,self).__init__()
        self.ui = Ui_DataSelect()
        self.ui.setupUi(self)
        self.parent = parent
        self.FNs = []
        # attach buttons
        self.ui.pb_SetSelect.clicked.connect(self.setSelect)
        self.ui.pb_SelectNifti.clicked.connect(self.selectNifti)
        
    def selectNifti(self):
        try:
            self.ui.list_files.clear()
            # get file directory
            w = QtWidgets.QWidget()
            filters = "NIFTI files (*.nii)"
            selected_filter = filters
            full_path,used_filter = QtWidgets.QFileDialog.getOpenFileName(
                                            w, 'Select image file',
                                            self.parent.datadir,
                                            filters,selected_filter)
            print(full_path)
            
            if len(full_path)==0:
                return
            filedir,FN = os.path.split(full_path)
            FN,ext = os.path.splitext(FN)
            
            # generate file names
            self.parent.datadir = filedir
            imp_list = []  # create an empty list
            for dirName, subdirList, fileList in os.walk(filedir):
                for filename in fileList:
                    if ext in filename.lower():  # get all files of same extension
                        imp_list.append(os.path.join(dirName,filename))
            imp_list = natsorted(imp_list)
            
            FNs = []
            # add to list view
            for fname in imp_list:
                FNs.append(fname)
                _,fn = os.path.split(fname)
                item = QtWidgets.QListWidgetItem(fn)
                self.ui.list_files.addItem(item)
                
            self.FNs = FNs
            self.ui.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            self.ui.list_files.customContextMenuRequested.connect(self.contextMenu)
            
        except Exception as e:
            print(e)
    def contextMenu(self,position):
        try:
            menu = QtWidgets.QMenu()
            deleteAction = menu.addAction("Remove from import")
            action = menu.exec_(self.ui.list_files.mapToGlobal(position))
            if action == deleteAction:                        
                for item in self.ui.list_files.selectedItems():
                    self.ui.list_files.takeItem(self.ui.list_files.row(item))
                    FNs = []
                    for index in range(self.ui.list_files.count()):
                         FNs.append(self.ui.list_files.item(index).text())
                    self.FNs = FNs
        except Exception as e:
            print(e)
            
    def setSelect(self):
        self.parent.file_list = self.FNs
            
        self.parent.disp_msg = 'Files selected'
        self.parent.saved = False
        self.hide()
        self.parent.ImportImages()
        self.destroy()


#%%
class NiftiImportThread(QThread):
    images_sig = pyqtSignal(np.ndarray)
    errorsig = pyqtSignal()
    def __init__(self,FN):
        QThread.__init__(self)
        self.FN = FN

    def __del__(self):
        self.wait()
        
    def noise_elim(self,images):
        per99 = np.percentile(images,99.5)
        if np.max(images)>2*per99:
            filt_ims = median_filter(images,size=(3,3,3))
            use_mean = .5*per99
            mask = np.abs(images-filt_ims)>use_mean
            images[mask] = filt_ims[mask]
        return images

    def run(self):
        try:
            # import water nifti
            nft = nib.load(self.FN)
            
            # adjust orientation
            canon_nft = nib.as_closest_canonical(nft)
            
            ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
            ims_send = self.noise_elim(ims)
            self.images_sig.emit(ims_send)
                
        except Exception as e:
            self.errorsig.emit()
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            self.quit()
#%%
class ModelLoadThread(QThread):
    model_sig = pyqtSignal(keras.engine.training.Model)
    error_sig = pyqtSignal(str)
    def __init__(self,filename,graph):
#        QThread.__init__(self)
        super(ModelLoadThread,self).__init__()
        self.filename = filename
        self.graph = graph
    def __del__(self):
        print('thread deleted')

    def run(self):
        try:
            # load model
            with self.graph.as_default():
                model = keras.models.load_model(self.filename)
            self.model_sig.emit(model)
        except Exception as e:
            print(e)
            self.error_sig.emit(e)
            print('thread error')
            self.quit()
#%%
class SegThread(QThread):
    segmask_sig = pyqtSignal(np.ndarray)
    error_sig = pyqtSignal(str)
    batch_sig = pyqtSignal(int)
    
    def __init__(self,graph,inputs,model,numB):
        QThread.__init__(self)
        self.graph = graph
        self.inputs = inputs
        self.model = model
        self.numB = numB
    def __del__(self):
        self.wait()

    def run(self):
        try:
            chunked_inputs = np.array_split(self.inputs,self.numB)
            output = np.zeros(self.inputs.shape)
            bnum = 0
            with self.graph.as_default():
                for inpchunk in chunked_inputs:
                    bsize = inpchunk.shape[0]
                    output[bnum:bnum+bsize,...] = self.model.predict_on_batch(inpchunk)
                    bnum += bsize
                    self.batch_sig.emit(bsize)
                    
            self.segmask_sig.emit(output[...,0])
        except Exception as e:
            print(e)
            self.error_sig.emit(e)
            self.quit()
#%%
class CalcThread(QThread):
    calcs_sig = pyqtSignal(tuple)
    error_sig = pyqtSignal(str)
    
    def __init__(self,voxvol,uncor_segmask,segmask):
        QThread.__init__(self)
        self.voxvol = voxvol
        self.uncor_segmask = uncor_segmask
        self.segmask = segmask
    def __del__(self):
        self.wait()

    def run(self):
        try:
            OTV = 1e-3*np.sum(self.uncor_segmask)*self.voxvol
            CTV = 1e-3*np.sum(self.segmask)*self.voxvol
            CUV = 1e-3*np.sum(np.multiply(self.segmask,
                                         (1-self.uncor_segmask)))*self.voxvol
            OUV = 1e-3*np.sum(np.multiply(self.uncor_segmask,
                                                   (1-self.segmask)))*self.voxvol
            if CTV!=0:
                diff = 100*(OUV+CUV)/CTV
            else:
                diff = 0
                
            intersect = np.sum(np.multiply(self.segmask,self.uncor_segmask))
            if (OTV+CTV)!=0:
                dice = 200*(intersect*1e-3*self.voxvol)/(OTV+CTV)
            else:
                dice = 0
            
            calcs = (OTV, CTV, OUV, CUV,diff,dice)
            self.calcs_sig.emit(calcs)
        except Exception as e:
            print(e)
            self.error_sig.emit(e)
            self.quit()           
#%%
if __name__ == "__main__":
    app = 0
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        print('rerunning')
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
    print('here')