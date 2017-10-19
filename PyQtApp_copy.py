import sys
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import pyqtProperty,pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor
import scipy.io as spio
import pyqtgraph as pg
import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.morphology import binary_fill_holes
import os
import psutil
import configparser
import dicom
from natsort import natsorted
import keras
import h5py
from CustomMetrics import jac_met, dice_coef, dice_coef_loss

pg.setConfigOptions(imageAxisOrder='row-major')

pyqtd_main= "pyqt_main.ui"
pyqtd_imp = "pyqt_import.ui"
pyqtd_cor = "pyqt_corrector.ui"
pyqtd_prefs = "pyqt_prefs.ui"
pyqtd_slider_dialog = "pyqt_sliderdialog.ui"

Ui_MainWindow, QtBaseClass1 = uic.loadUiType(pyqtd_main)
Ui_ImpWindow, QtBaseClass2 = uic.loadUiType(pyqtd_imp)
Ui_CorWindow, QtBaseClass3 = uic.loadUiType(pyqtd_cor)
Ui_PrefsWindow, QtBaseClass4 = uic.loadUiType(pyqtd_prefs)
Ui_SliderDialog, QtBaseClass5 = uic.loadUiType(pyqtd_slider_dialog)
 
class MainApp(QtBaseClass1, Ui_MainWindow):
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
        self.ui.actionEdit_Preferences.triggered.connect(self.setPreferences)
        
        self.ui.actionShift_Dimensions.triggered.connect(self.shiftDims)
        self.ui.actionPrepare_Inputs.triggered.connect(self.prepInputs)
        self.ui.actionSegment.triggered.connect(self.segment)
        self.ui.actionCorrect_Segmentation.triggered.connect(self.openCorrector)
        self.ui.actionCalculate_PDWF.triggered.connect(self.calcPDWF)
        
        self.ui.actionOpen_User_Manual.triggered.connect(self.openHelp)
        
        # initialize some variables
        self.configFN = 'config.ini'
        self.graph = keras.backend.tf.get_default_graph()
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
        self.SIbounds = np.array([])
        self.fov = np.array([22,22,36])
        self.spatres = []
        self.maskdisp = False
        self.bounddisp = False
        self.saved = True
        self.WLmult = .5
        self.mask_alpha = .4
        self.segThresh = .5
        self.cursel = []
        
        # initialize display message
        self.disp_msg = 'Welcome to PDWFnet'
        
        # Initialize or load config file
        if os.path.isfile(self.configFN):
            config = configparser.ConfigParser()
            try:
                config.read(self.configFN);
            except Exception as e:
                self.error_msg.set('Unable to read config file. Creating new')
                config = configparser.ConfigParser()
                curdir = os.getcwd()
                fovstr = ",".join(str(e) for e in self.fov)
                alpha = str(self.mask_alpha)
                thresh = str(self.PDWFthresh)
                config['DEFAULT'] = {'MAT Directory': curdir,
                          'DCM Directory': curdir,
                          'Model directory': curdir,
                          'Save Directory': curdir,
                          'FOV': fovstr,
                          'MaskAlpha': alpha,
                          'SegThresh': .5,
                          'PDWFthresh': thresh}
                with open(self.configFN, 'w') as configfile:
                    config.write(configfile)
        else:
            config = configparser.ConfigParser()
            curdir = os.getcwd()
            fovstr = ",".join(str(e) for e in self.fov)
            alpha = str(self.mask_alpha)
            thresh = str(self.PDWFthresh)
            config['DEFAULT'] = {'MAT Directory': curdir,
                                 'DCM Directory': curdir,
                                 'Model directory': curdir,
                                 'Save Directory': curdir,
                                 'FOV': fovstr,
                                 'MaskAlpha': alpha,
                                 'SegThresh': .5,
                                 'PDWFthresh': thresh}
            with open(self.configFN, 'w') as configfile:
                config.write(configfile)
                
        # parse config into self
        self.config = config
        self.fov = np.array([float(i) for i in self.config.get('DEFAULT','fov').split(",")])
        self.mask_alpha = np.float16(self.config.get('DEFAULT','MaskAlpha'))
        self.segThresh = np.float16(self.config.get('DEFAULT','SegThresh'))
        self.PDWFthresh = np.float16(self.config.get('DEFAULT','PDWFthresh'))
        self.matdir = self.config.get('DEFAULT','mat directory')
        self.dcmdir = self.config.get('DEFAULT','dcm directory')
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
        self.segmask = []
        self.segOutput = []
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
                self.savedir,_ = os.path.split(data_path)
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
            suggest = os.path.join(self.savedir,'SegData.mat')
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
                a['SIbounds'] = self.SIbounds
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
        self.ui.menubar.setEnabled(False)
        self.ui.progBar.setRange(0,0)
        self.ui.progBar.setVisible(True)
        QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        self.load_thread = ModelLoadThread(data_path,self.graph)
        self.load_thread.finished.connect(self.loadModelFinish)
        self.load_thread.model_sig.connect(self.loadModelModel)
        self.load_thread.error_sig.connect(self.loadModelError)
        
        self.load_thread.start()
               
        
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
        if len(self.inputs)==0:
            self.error_msg = 'Inputs not prepared'
            return
        try:
            # calculate batch sizes
            avail_mem = psutil.virtual_memory().available
            inpshape = self.inputs.shape
            needed_mem_per_batch = inpshape[1]*inpshape[2]*inpshape[3]*885
            batch_size_allowed = np.floor(.5*avail_mem/needed_mem_per_batch)
            num_batch = np.ceil(inpshape[0]/batch_size_allowed)
                
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
    def segFinish(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.menubar.setEnabled(True)
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        try:
            self.maskOn()
            self.maskdisp = True
            self.resetView()
        except Exception as e:
            print(e)
    def segBatch(self,addNum):
        self.ui.progBar.setRange(0,self.inputs.shape[0])
        self.ui.progBar.setValue(self.ui.progBar.value()+addNum)
    def segGotMask(self,output):
        self.disp_msg = 'Segmentation complete'
        self.segOutput = output
        self.segmask = np.float16(self.segOutput>self.segThresh)
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
            self.segmask = np.float16(self.segOutput>self.segThresh)
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
        if len(self.imagesW)==0:
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
            
            # Add initial image
            imshape = np.array(self.imagesW.shape,dtype=np.int16)
            midinds = np.round(imshape/2).astype(np.int16)
            self.img_item_ax.setImage(self.curimages[midinds[0],...])
            self.img_item_cor.setImage(self.curimages[:,midinds[1],:])
            self.img_item_sag.setImage(self.curimages[:,:,midinds[2]])
            
            # add masks, even if empty
            msksiz = np.r_[self.imagesW.shape,4]
            msk = np.zeros(msksiz,dtype=float)
            msk[...,0] = 1
            msk[...,1] = .4
            msk[...,2] = .4
            if not len(self.segmask)==0:
                msk[...,3] = self.mask_alpha*self.segmask.astype(float)
                self.maskdisp = True
            self.mask = msk
            self.msk_item_ax.setImage(self.mask[midinds[0],...])
            self.msk_item_cor.setImage(self.mask[:,midinds[1],...])
            self.msk_item_sag.setImage(self.mask[:,:,midinds[2],...])
            
            # Create plot items
            # S/I boundaries
            SIpen = pg.mkPen(color=(0, 50, 255, 120),width=2)
            SIbounds = np.array([1,imshape[0]-1],dtype=np.int16)
            self.SIbounds = SIbounds
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
            
            #calculate buffer
            buff = 15 #in mm
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
            
            # Setup indices
            self.inds = midinds
            self.volshape = imshape
                        
            # attach callbacks
            self.ui.slideAx.valueChanged.connect(self.axSlide)
            
            self.ui.viewAxial.wheelEvent = self.axScroll
            self.img_item_ax.mousePressEvent = self.axClickEvent
            
            self.ui.viewCoronal.wheelEvent = self.corScroll
            self.img_item_cor.mouseClickEvent = self.corClickEvent
            self.img_item_cor.mouseDragEvent = self.corDragEvent
            
            self.ui.viewSag.wheelEvent = self.sagScroll
            self.img_item_sag.mouseClickEvent = self.sagClickEvent
            self.img_item_sag.mouseDragEvent = self.sagDragEvent
            
            self.ui.viewCoronal.scene().setMoveDistance(3)
            self.ui.viewSag.scene().setMoveDistance(3)
            # attach context menus
            self.ui.viewAxial.customContextMenuRequested.connect(self.axCMenu)
            
            # make full screen
#            self.showMaximized()
            
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
        try:
            if ev.button()==1:
                self.img_item_ax.mouseMoveEvent = self.axMoveEvent
                self.img_item_ax.mouseReleaseEvent = self.axReleaseEvent
                posx = ev.pos().x()
                posy = ev.pos().y()
                self.axMove(posx,posy)
                ev.accept()
            elif ev.button()==4:
                self.cursel='ax'
                self.img_item_ax.mouseMoveEvent = self.LevelEvent
                self.img_item_ax.mouseReleaseEvent = self.axReleaseEvent
                self.prevLevel = np.array(self.img_item_ax.getLevels())
                self.startPos = np.array([ev.pos().x(),ev.pos().y()])
                ev.accept()
            else:
                ev.ignore()
        except Exception as e:
            print(e)
    def axMoveEvent(self,ev):
        posx = ev.pos().x()
        posy = ev.pos().y()
        self.axMove(posx,posy)
        ev.accept()
    def axReleaseEvent(self,ev):
        self.img_item_ax.mouseMoveEvent = ''
        ev.accept()
    
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
        menu.addAction(watimAction)
        menu.addAction(fatimAction)
        menu.addSeparator()
        togmaskAction = QtWidgets.QAction("Display mask", menu, checkable=True)
        togmaskAction.setChecked(self.maskdisp)
        menu.addAction(togmaskAction)
        togboundsAction = QtWidgets.QAction("Display SI bounds", menu, checkable=True)
        togboundsAction.setChecked(self.bounddisp)
        menu.addAction(togboundsAction)
        segthreshAction = menu.addAction("Adjust Segmentation Threshold...")
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
                self.line_cor_S.setData(x = np.array([0,1]),
                                         y = np.array([self.SIbounds[1],self.SIbounds[1]]))
                self.line_cor_I.setData(x = np.array([0,1]),
                                         y = np.array([self.SIbounds[0],self.SIbounds[0]]))
                self.line_sag_S.setData(x = np.array([0,1]),
                                         y = np.array([self.SIbounds[1],self.SIbounds[1]]))
                self.line_sag_I.setData(x = np.array([0,1]),
                                         y = np.array([self.SIbounds[0],self.SIbounds[0]]))
            else:
                self.bounddisp = True
                self.line_cor_S.setData(x = np.array([0,self.volshape[2]]),
                                         y = np.array([self.SIbounds[1],self.SIbounds[1]]))
                self.line_cor_I.setData(x = np.array([0,self.volshape[2]]),
                                         y = np.array([self.SIbounds[0],self.SIbounds[0]]))
                self.line_sag_S.setData(x = np.array([0,self.volshape[1]]),
                                         y = np.array([self.SIbounds[1],self.SIbounds[1]]))
                self.line_sag_I.setData(x = np.array([0,self.volshape[1]]),
                                         y = np.array([self.SIbounds[0],self.SIbounds[0]]))
                
        elif action == segthreshAction:
            self.setSegThresh()
        elif action == resetAction:
            self.resetView()
    def maskOn(self):
        # show mask
        try:
            self.mask[...,3] = .3*self.segmask.astype(float)
        except:
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
            
    def resetView(self):
        # reset viewbox ranges
        self.vbox_ax.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[1]))
        self.vbox_cor.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[0]))
        self.vbox_sag.setRange(xRange=(0,self.volshape[1]),yRange=(0,self.volshape[0]))
        # remake mask
        if not len(self.segOutput)==0:
            self.segmask = np.float16(self.segOutput>self.segThresh)
            self.mask[...,3] = self.mask_alpha*self.segmask
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
        
    def prepInputs(self):
        self.disp_msg = 'Preparing inputs for segmentation...'
        if len(self.imagesW)==0 or len(self.imagesF)==0:
            self.error_msg = "No data loaded"
            return
        try:
            # normalize
            imagesW = np.float16(self.imagesW)
            imagesF = np.float16(self.imagesF)
            for ss in range(imagesW.shape[0]):
                np.divide(imagesW[ss,:,:],np.max(imagesW[ss,:,:]),out=imagesW[ss,:,:])
                np.divide(imagesF[ss,:,:],np.max(imagesF[ss,:,:]),out=imagesF[ss,:,:])
                
            # concatenate channels
            self.inputs = np.stack((imagesW,imagesF),axis=3)
    
            self.disp_msg = "Input image preparation complete"
        except Exception as e:
            print(e)
    def calcPDWF(self):
        try:
            self.disp_msg = 'Calculating PDWF value...'
            if len(self.PDWFmap)==0:
                self.error_msg = "No PDWF map"
                return
            if  len(self.segmask)==0:
                self.error_msg = 'No Segmentation Mask'
                return
            PDWFmask = np.multiply(self.PDWFmap,self.segmask.astype(float))
            PDWFmask[PDWFmask>self.PDWFthresh] = self.PDWFthresh
            PDWFmask[PDWFmask<-self.PDWFthresh] = -self.PDWFthresh
            voxvol = np.prod(self.spatres)
            water_vol = np.abs(np.sum(PDWFmask))*voxvol
            tot_vol = np.sum(self.segmask.astype(float))*voxvol
            self.PDWF = water_vol/tot_vol
            self.disp_msg = "Water volume is {:.3f} cc".format(water_vol/1000)
            self.disp_msg = "Total volume is {:.3f} cc".format(tot_vol/1000)
            self.disp_msg = "PDWF is {:.2%}".format(self.PDWF)
        except Exception as e:
            print(e)
            
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
        self.config['DEFAULT']['dcm directory'] = self.dcmdir
        self.config['DEFAULT']['model directory'] = self.mdldir
        self.config['DEFAULT']['save directory'] = self.savedir
        self.config['DEFAULT']['fov'] = ','.join(str(x) for x in self.fov)
        self.config['DEFAULT']['SegThresh'] = str(self.segThresh)
        self.config['DEFAULT']['MaskAlpha'] = str(self.mask_alpha)
        self.config['DEFAULT']['PDWFthresh'] = str(self.PDWFthresh)
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
        self.destroy()
#%%
class ThresholdSlider(QtBaseClass5,Ui_SliderDialog):
    def __init__(self,parent):
        try:
            super(ThresholdSlider,self).__init__()
            self.ui = Ui_SliderDialog()
            self.ui.setupUi(self)
            loc = QtWidgets.QDesktopWidget().screenGeometry().topRight()/2
            self.move(loc.x(),loc.y())
            self.parent = parent
            self.ui.slider.setValue(self.parent.segThresh*100)
            self.ui.lcd.display(self.ui.slider.value()/100)
            
            # attach callbacks
            self.ui.slider.valueChanged.connect(self.sliderChanged)
        except Exception as e:
             print(e)
             
    def sliderChanged(self,value):
         self.ui.lcd.display(value/100)
         self.parent.segThresh = np.float16(value/100)
         self.parent.updateMaskThresh()
         
    def accept(self):
        try:
            print('Accepted')
            self.destroy()
        except Exception as e:
            print(e)
    def reject(self):
        try:
            print('Rejected')
            self.destroy()
        except Exception as e:
            print(e)
#%%
class PrefsWindow(QtBaseClass4,Ui_PrefsWindow):
    def __init__(self,parent):
        try:
            super(PrefsWindow,self).__init__()
            self.ui = Ui_PrefsWindow()
            self.ui.setupUi(self)
            self.parent = parent
            self.prefs_saved = True
            
            self.ui.stackDisplay.setCurrentIndex(0)
            # attach buttons
            self.ui.buttonFOV.clicked.connect(lambda ev: self.setPage(ev, 0))
            self.ui.buttonDisplay.clicked.connect(lambda ev: self.setPage(ev, 1))
            self.ui.buttonPDWF.clicked.connect(lambda ev: self.setPage(ev, 2))
            self.ui.buttonOther.clicked.connect(lambda ev: self.setPage(ev, 3))
            
            self.accepted.connect(self.accept)
            self.rejected.connect(self.reject)
        
            # distribute pref data
            self.ui.boxFOVz.setValue(self.parent.fov[0])
            self.ui.boxFOVy.setValue(self.parent.fov[1])
            self.ui.boxFOVx.setValue(self.parent.fov[2])
            
            self.ui.slideAlpha.setValue(np.round(self.parent.mask_alpha*20).astype(np.int16))
            self.ui.lcdAlpha.display(self.ui.slideAlpha.value()/20)
            
            # attach callbacks
            self.ui.slideAlpha.valueChanged.connect(self.slideAlphaChanged)
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
    def savePrefs(self):
        print('saving prefs')
        FOVx = np.array([self.ui.boxFOVx.value()])
        FOVy = np.array([self.ui.boxFOVy.value()])
        FOVz = np.array([self.ui.boxFOVz.value()])
        self.parent.fov = np.r_[FOVz,FOVy,FOVx]
        
        self.parent.mask_alpha = self.ui.lcdAlpha.value()
        
        self.parent.saveConfig()
        self.parent.disp_msg = 'Preferences updated'
        
    def setPage(self,ev,pnum):
        self.ui.stackDisplay.setCurrentIndex(pnum)
    def slideAlphaChanged(self,ev):
        val = ev/20
        self.ui.lcdAlpha.display(val)
        self.prefs_saved = False
    def accept(self):
        try:
            print('Accepted')
            self.savePrefs()
            self.destroy()
        except Exception as e:
            print(e)
    def reject(self):
        try:
            print('Rejected')
            self.destroy()
        except Exception as e:
            print(e)
    def closeEvent(self, ev):
        # check for saved
        try:
            print('tryna close')
            if not self.prefs_saved:
                quit_msg = "Apply changes?"
                reply = QtWidgets.QMessageBox.warning(self, 'Closing', 
                                 quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            
                if reply == QtWidgets.QMessageBox.Yes:
                    ev.accept()
                    self.savePrefs()
                    self.destroy()
                else:
                    ev.accept()
                    self.destroy()
            else:
                ev.accept()
        except Exception as e:
            print(e)
#%%
class ImpWindow(QtBaseClass2,Ui_ImpWindow):
    def __init__(self,parent):
        super(ImpWindow,self).__init__()
        self.ui = Ui_ImpWindow()
        self.ui.setupUi(self)
        self.parent = parent
        self.wFNs = []
        self.fFNs = []
        # attach buttons
        self.ui.matButton.clicked.connect(self.pick_mat)
        self.ui.impButton.clicked.connect(self.imp)
        self.ui.dcmButtonW.clicked.connect(self.pick_dcm)
        self.ui.dcmButtonF.clicked.connect(self.pick_dcm)
        
    def pick_mat(self):
        self.ui.listW.clear()
        self.ui.listF.clear()
        w = QtWidgets.QWidget()
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(w, 'Select MAT file',
                                                         self.parent.matdir,
                                                           'MAT files (*.mat)',)
        self.mat = spio.loadmat(filename,squeeze_me=True)
        if len(filename) == 0:
            return
        fdir,FN = os.path.split(filename)
        self.parent.filename = FN
        self.parent.matdir = fdir
        self.filename = filename
        varbls = spio.whosmat(filename)
        varbl_nms = [f[0] for f in varbls]
        for vname in varbl_nms:
            item = QtWidgets.QListWidgetItem(vname)
            self.ui.listW.addItem(item)
            item = QtWidgets.QListWidgetItem(vname)
            self.ui.listF.addItem(item)
        self.imptype = "mat"
        
    def pick_dcm(self):
        self.ui.listF.clear()
        # get file path
        w = QtWidgets.QWidget()
        filters = "DICOM files (*.dcm);;All files (*.*)"
        full_path,_ = QtWidgets.QFileDialog.getOpenFileName(w, 'Select DCM file',
                                                         self.parent.dcmdir,
                                                         filters)
        if len(full_path)==0:
            return
        fdir,fFN = os.path.split(full_path)
        self.parent.dcmdir,_ = os.path.split(fdir)
        FN,ext = os.path.splitext(fFN)
        self.ui.progressBar.setRange(0,0)
        # generate file names
        dcm_list = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(fdir):
            for filename in fileList:
                if ext in filename.lower():  # check whether the file's DICOM
                    dcm_list.append(os.path.join(dirName,filename))
        dcm_list = natsorted(dcm_list)
        # determine fat or water
        if self.sender().objectName()=='dcmButtonW':
            self.ui.listW.clear()
            wFNs = []
            for fname in dcm_list:
                wFNs.append(fname)
                _,fn = os.path.split(fname)
                item = QtWidgets.QListWidgetItem(fn)
                self.ui.listW.addItem(item)
            self.wFNs = wFNs
        elif self.sender().objectName()=='dcmButtonF':
            self.ui.listF.clear()
            fFNs = []
            for fname in dcm_list:
                fFNs.append(fname)
                _,fn = os.path.split(fname)
                item = QtWidgets.QListWidgetItem(fn)
                self.ui.listF.addItem(item)
            self.fFNs = fFNs
        if len(self.fFNs)!=0 and len(self.wFNs)!=0:
            if len(self.fFNs)!=len(self.wFNs):
                self.parent.error_msg = 'Warning: unequal number of images selected'
        self.imptype = "dcm"
        self.ui.progressBar.setRange(0,1)
        
    def imp(self):
        try:
            if self.imptype == "mat":
                self.parent.disp_msg = 'Loading...'
                self.ui.progressBar.setRange(0,0)
                # get listbox selections
                fieldW = self.ui.listW.currentItem().text()
                fieldF = self.ui.listF.currentItem().text()
                self.imp_thread = MatImportThread(self.filename,fieldW,fieldF)
                self.imp_thread.finished.connect(self.imp_finish_mat)
                self.imp_thread.imagesWsig.connect(self.gotImagesW)
                self.imp_thread.imagesFsig.connect(self.gotImagesF)
                self.imp_thread.PDWFsig.connect(self.gotPDWF)
                self.imp_thread.imDimsig.connect(self.ErrorDim)
                self.imp_thread.errorsig.connect(self.matError)
                
                self.imp_thread.start()
                
            elif self.imptype == "dcm":
                self.parent.disp_msg = "Importing dicoms..."
                num = len(self.wFNs)+len(self.fFNs)
                self.ui.progressBar.setRange(0,num)
                self.imp_thread = DcmImportThread(self.wFNs,self.fFNs)
                self.imp_thread.finished.connect(self.imp_finish_dcm)
                self.imp_thread.imDimsig.connect(self.ErrorDim)
                self.imp_thread.slicesig.connect(self.dcmSlice)
                self.imp_thread.imagesWsig.connect(self.gotImagesWdcm)
                self.imp_thread.imagesFsig.connect(self.gotImagesFdcm)
                self.imp_thread.PDWFsig.connect(self.gotPDWF)
                self.imp_thread.infosig.connect(self.gotDcmInfo)
                self.imp_thread.errorsig.connect(self.dcmError)
                self.imp_thread.start()
            else:
                self.parent.error_msg.set("Choose files to import")
        except Exception as e:
            print(e)
            
    def gotImagesWdcm(self,imagesW):
        self.parent.imagesW = imagesW
    def gotImagesFdcm(self,imagesF):
        self.parent.imagesF = imagesF
    def dcmSlice(self,add):
        self.ui.progressBar.setValue(self.ui.progressBar.value()+add)
    def gotDcmInfo(self,info):
        self.parent.spatres = info['SpatRes']
    def dcmError(self):
        self.parent.error_msg = 'Error importing DICOMS'
    def imp_finish_dcm(self):
        self.parent.curimages = self.parent.imagesW
        self.parent.disp_msg = 'DICOMs imported'
        self.parent.saved = False
        self.hide()
        self.parent.InitDisplay()
        self.destroy()
        
    def gotImagesW(self,imagesW):
        self.parent.imagesW = imagesW
        self.ui.progressBar.setRange(0,2)
        self.ui.progressBar.setValue(1)
    def gotImagesF(self,imagesF):
        self.parent.imagesF = imagesF
        self.ui.progressBar.setValue(2)
    def gotPDWF(self,PDWF):
        self.parent.PDWFmap = PDWF
    def ErrorDim(self):
        self.parent.error_msg = 'Image dimensions do not match'
    def matError(self):
        self.parent.error_msg = 'Error loading MAT file'
    def imp_finish_mat(self):
        self.parent.curimages = self.parent.imagesW
        # calc estimated spatial resolution
        self.parent.spatres = self.parent.fov/self.parent.imagesW.shape*10

        self.parent.disp_msg = "{} loaded".format(self.parent.filename)
        self.ui.progressBar.setRange(0,1)
        self.parent.saved = False
        self.hide()
        self.parent.InitDisplay()
        self.destroy()
#%%
class CorWindow(QtBaseClass3,Ui_CorWindow):
    def __init__(self,parent,levs,ind):
        try:
            super(CorWindow,self).__init__()
            self.ui = Ui_CorWindow()
            self.ui.setupUi(self)
            self.parent = parent
            self.images = parent.imagesW
            self.mask = np.copy(parent.mask)
            if len(parent.segmask)==0:
                self.segmask = np.zeros(self.images.shape)
            else:
                self.segmask = np.copy(parent.segmask)
                
            self.volshape = self.images.shape
            self.WLmult = parent.WLmult
            self.alph = self.parent.mask_alpha
            self.cor_saved = True
            # setup plots
            asprat1 = self.parent.spatres[1]/self.parent.spatres[2]
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
            self.ind = ind
            self.img_item.setImage(self.images[self.ind,...])
            self.img_item.setLevels(levs)
            self.msk_item.setImage(self.mask[self.ind,...])
            
            #adjust view range
            self.vbox.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[1]),
                                  padding=0.,disableAutoRange=True)
            
            # create starting brush            
            self.BrushMult = 1
            self.brush = self.my_brush_mask(self.ui.slideBrushSize.value()*self.BrushMult)
            self.rad = self.ui.slideBrushSize.value()
            
            # create brush cursor
            self.dot = pg.ScatterPlotItem(x=np.array([5.]),y=np.array([5.]),
                                    symbol='o',symbolSize=self.rad,symbolPen=(100,100,100,.5),
                                    brush=None,pxMode=False)
            self.vbox.addItem(self.dot)
            pg.SignalProxy(self.ui.viewAxial.scene().sigMouseMoved, rateLimit=100, slot=self.mouseMoved)
            self.ui.viewAxial.scene().sigMouseMoved.connect(self.mouseMoved)
#            self.ui.viewAxial.scene().sigMouseHover.connect(self.brushEnterExit)
            
            # Setup slider
            self.ui.slideAx.setMaximum(self.images.shape[0]-1)
            self.ui.slideAx.setValue(self.ind)
            
            # attach callbacks
            self.ui.slideAx.valueChanged.connect(self.slide)
            self.ui.slideAlpha.valueChanged.connect(self.alphaSlide)
            self.ui.slideBrushSize.valueChanged.connect(self.brushSlide)
            self.ui.viewAxial.wheelEvent = self.scroll
            self.ui.actionSave.triggered.connect(self.saveCor)
            self.ui.actionReset_View.triggered.connect(self.resetView)
            self.img_item.mousePressEvent = self.clickEvent
            self.img_item.hoverEvent = self.brushHover
            self.msk_item.mousePressEvent = self.clickEvent
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
    def slide(self,ev):
        self.img_item.setImage(self.images[ev,...],autoLevels=False)
        self.msk_item.setImage(self.mask[ev,...])
        self.ind = ev
    
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
        except Exception as e:
            print(e)
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
        try:
            if ev.isEnter():
                print('enter')
                QtWidgets.QApplication.setOverrideCursor(QCursor(Qt.BlankCursor))
                self.dot.setData(symbolPen=(100,100,100,0.5))
                ev.accept()
            elif ev.isExit():
                print('exit')
                QtWidgets.QApplication.restoreOverrideCursor()
                self.dot.setData(symbolPen=(100,100,100,0))
                ev.accept()
            else:
                ev.ignore()
        except Exception as e:
            pass
    def clickEvent(self,ev):
        if ev.button()==1 or ev.button()==2:
            self.img_item.mouseMoveEvent = self.movingEvent
            self.msk_item.mouseMoveEvent = self.movingEvent
            self.img_item.mouseReleaseEvent = self.releaseEvent
            self.msk_item.mouseReleaseEvent = self.releaseEvent
            self.curmask = self.segmask[self.ind,...]
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
            self.cor_saved = False
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
        
    def saveCor(self,ev):
        try:
            if not ev:
                self.parent.mask = np.copy(self.mask)
                self.parent.segmask = np.copy(self.segmask)
                self.parent.updateIms()
                self.parent.disp_msg = 'Corrections applied'
                self.cor_saved = True
                self.parent.saved = False
        except Exception as e:
            print(e)
            
    @pyqtProperty(bool)
    def cor_saved(self):
            return self._cor_saved
    @cor_saved.setter
    def cor_saved(self, value):
        self._cor_saved = value
        if value:
            self.ui.actionSave.setEnabled(False)
        else:
            self.ui.actionSave.setEnabled(True)
        QtWidgets.qApp.processEvents()
        
    def closeEvent(self, ev):
        # check for saved
        try:
            if not self.cor_saved:
                quit_msg = "Corrections have not been applied. Are you sure you wish to close?"
                reply = QtWidgets.QMessageBox.warning(self, 'Closing', 
                                 quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            
                if reply == QtWidgets.QMessageBox.Yes:
                    ev.accept()
                    self.destroy()
                else:
                    ev.ignore()
                    return
            else:
                ev.accept()
        except Exception as e:
            print(e)
#%%
class MatImportThread(QThread):
    imagesWsig = pyqtSignal(np.ndarray)
    imagesFsig = pyqtSignal(np.ndarray)
    PDWFsig = pyqtSignal(np.ndarray)
    errorsig = pyqtSignal()
    imDimsig = pyqtSignal()
    def __init__(self,filename,fieldW,fieldF):
        QThread.__init__(self)
        self.filename = filename
        self.fieldW = fieldW
        self.fieldF = fieldF

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
            # import images
            matW = spio.loadmat(self.filename,variable_names=[self.fieldW])
    
            # determine slice axis and roll to fit, eliminate noise
            # water images
            imagesW = np.abs(matW[self.fieldW])
            wimshape = imagesW.shape
            imagesW = np.rollaxis(imagesW,np.argmin(imagesW.shape),0)
            imagesW = self.noise_elim(imagesW)
            self.imagesWsig.emit(imagesW)
            # fat images
            matF = spio.loadmat(self.filename,variable_names=[self.fieldF])
            imagesF = np.abs(matF[self.fieldF])
            if not np.array_equal(wimshape,imagesF.shape):
                self.imDimsig.emit()
                return
            imagesF = np.rollaxis(imagesF,np.argmin(imagesF.shape),0)
            self.imagesF = self.noise_elim(imagesF)
            self.imagesFsig.emit(imagesF)
            
            # calculate PDWF map
            with np.errstate(divide='ignore',invalid='ignore'):
                PDWFmap = np.divide(imagesW,imagesW+imagesF)
            PDWFmap = np.nan_to_num(PDWFmap)
            self.PDWFsig.emit(PDWFmap)
        except Exception as e:
            self.errorsig.emit()
            print(e)
#%%
class DcmImportThread(QThread):
    slicesig = pyqtSignal(int)
    imagesWsig = pyqtSignal(np.ndarray)
    imagesFsig = pyqtSignal(np.ndarray)
    PDWFsig = pyqtSignal(np.ndarray)
    infosig = pyqtSignal(dict)
    imDimsig = pyqtSignal()
    errorsig = pyqtSignal()
    def __init__(self,wFNs,fFNs):
        QThread.__init__(self)
        self.wFNs = wFNs
        self.fFNs = fFNs

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
            # Get ref file
            RefDs = dicom.read_file(self.wFNs[0])
            # Load dimensions based on the number of rows, columns, and slices
            volsize = (int(RefDs.Rows), int(RefDs.Columns), len(self.wFNs))
            # Load spacing values (in mm)
            spatres = np.array([float(RefDs.SpacingBetweenSlices), float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1])])
            # Put into dict
            info ={'SpatRes':spatres}
            self.infosig.emit(info)
            # The array is sized based on volsize
            wims = np.zeros(volsize, dtype=RefDs.pixel_array.dtype)
            # loop through all the DICOM files
            for filenameDCM in self.wFNs:
                # read the file
                ds = dicom.read_file(filenameDCM)
                # store the raw image data
                wims[:, :, self.wFNs.index(filenameDCM)] = np.abs(ds.pixel_array)
                self.slicesig.emit(1)

            # adjust orientation
            des_orient = np.array([[1,0],[0,1]])
            orient_vec = np.array(np.round(RefDs.ImageOrientationPatient))
            act_orient = np.stack((orient_vec[0:2],orient_vec[3:5]))
            rotmat90 = np.array([[0,-1],[1,0]])
            numrot = 0
            isrot = np.array_equal(des_orient,act_orient)
            while not isrot:
                act_orient = np.matmul(rotmat90,act_orient)
                numrot += 1
                isrot = np.array_equal(des_orient,act_orient)

            wims = np.rot90(wims,-numrot)
            wims = np.rollaxis(wims,2,0)
            wims_send = self.noise_elim(wims)
            self.imagesWsig.emit(wims_send)

             # Get ref file
            RefDs = dicom.read_file(self.fFNs[0])

            # Load dimensions based on the number of rows, columns, and slices
            volsizeF = (int(RefDs.Rows), int(RefDs.Columns), len(self.wFNs))
            if not np.array_equal(volsize,volsizeF):
                self.imDimsig.emit()
                return

            # The array is sized based on volsize
            fims = np.zeros(volsizeF, dtype=RefDs.pixel_array.dtype)

            # loop through all the DICOM files
            for filenameDCM in self.fFNs:
                # read the file
                ds = dicom.read_file(filenameDCM)
                # store the raw image data
                fims[:, :, self.fFNs.index(filenameDCM)] = np.abs(ds.pixel_array)
                self.slicesig.emit(1)

            # adjust orientation
            orient_vec = np.array(np.round(RefDs.ImageOrientationPatient))
            act_orient = np.stack((orient_vec[0:2],orient_vec[3:5]))
            numrot = 0
            isrot = np.array_equal(des_orient,act_orient)
            while not isrot:
                act_orient = np.matmul(rotmat90,act_orient)
                numrot += 1
                isrot = np.array_equal(des_orient,act_orient)

            fims = np.rot90(fims,-numrot)
            fims = np.rollaxis(fims,2,0)
            fims_send = self.noise_elim(fims)
            self.imagesFsig.emit(fims_send)
            
            # calculate PDWF map
            with np.errstate(divide='ignore',invalid='ignore'):
                PDWFmap = np.divide(wims,wims+fims)
            PDWFmap = np.nan_to_num(PDWFmap)
            self.PDWFsig.emit(PDWFmap)
        except Exception as e:
            self.errorsig.emit()
            print(e)
#%%
class ModelLoadThread(QThread):
    model_sig = pyqtSignal(keras.engine.training.Model)
    error_sig = pyqtSignal(str)
    def __init__(self,filename,graph):
        QThread.__init__(self)
        self.filename = filename
        self.graph = graph
    def __del__(self):
        self.wait()

    def run(self):
        try:
            # load model
            with self.graph.as_default():
                model = keras.models.load_model(self.filename,
                               custom_objects={'jac_met':jac_met,
                                               'dice_coef':dice_coef,
                                               'dice_coef_loss':dice_coef_loss})
            self.model_sig.emit(model)
        except Exception as e:
            print(e)
            self.error_sig.emit(e)
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
if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    else:
        print('rerunning')
    window = MainApp()
    window.show()
    app.exec_()