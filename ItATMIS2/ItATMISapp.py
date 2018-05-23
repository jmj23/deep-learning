import sys
sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtCore import pyqtProperty,pyqtSignal, QThread, Qt
from PyQt5.QtGui import QCursor, QFont
import pyqtgraph as pg
import numpy as np
import json
from scipy.ndimage import median_filter
from scipy.ndimage.morphology import binary_fill_holes
# watershed methods
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.morphology import reconstruction

# Use first available GPU
import os
import GPUtil
try:
    if not 'DEVICE_ID' in locals():
        DEVICE_ID = GPUtil.getFirstAvailable()[0]
        print('Using GPU',DEVICE_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
except RuntimeError as e:
    print('No GPU available')
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
import configparser
import contextlib
import nibabel as nib
from natsort import natsorted
import keras
import h5py
import time
# Keras imports
from keras.layers import Input, Cropping2D, Conv2D, concatenate
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import ELU
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

pg.setConfigOptions(imageAxisOrder='row-major')

os.chdir('/home/jmj136/deep-learning/ItATMIS2')

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
            # self.ui.actionSaveModel.triggered.connect(self.saveCor)
            self.ui.actionReset_View.triggered.connect(self.resetView)
            self.ui.actionUndo.triggered.connect(self.undo)
            self.ui.actionUndo.setEnabled(False)
            self.ui.actionClear_Mask.triggered.connect(self.clearMask)
            self.ui.action_Save_Data.triggered.connect(self.data_save)
            self.ui.action_Load_Data.triggered.connect(self.data_load)
            self.ui.actionQuick_Select.triggered.connect(self.quick_select)
            self.ui.pb_SelectData.clicked.connect(self.DataSelect)
            self.ui.pb_Train.clicked.connect(self.Train)
            self.ui.pb_Evaluate.clicked.connect(self.EvalCurrentSubject)
            self.ui.listFiles.itemClicked.connect(self.FileListClick)
            self.ui.listFiles.itemDoubleClicked.connect(self.FileListSelect)
            
            # initialize display message
            self.disp_msg = 'Initializing...'
            
            # initialize some variables
            # configuration file name
            self.configFN = 'config.ini'
            # list of files to be included in processing
            self.file_list = []
            # list of indices of files that have masks created
            self.mask_list = []
            # current index in file list
            self.FNind = 0
            # most recently used directory of data
            self.datadir= []
            # current images loaded into app
            self.images = []
            # current segmentation mask
            self.segmask = []
            # list of undos
            self.segmask_undos = []
            # maximum number of undos
            self.max_undo = 10
            # current affine matrix
            self.niftiAff = []
            # current mask for current images
            self.mask = []
            # pre-segmentation
            self.WSseg = []
            # quick select starts as false
            self.qs_on = False
            # current slice indices
            self.inds= []
            # whether model is currently training
            self.training = False
            # number of classes to annotate
            self.numClasses = []
            # current set of targets
            self.targets = []
            # HDF5 file of annotations
            self.AnnotationFile = []
            # deep learning model
            self.model = []
            self.model_arch_path = []
            self.model_weights_path = []
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
            self.vbox = []
            # keras graph
            self.graph = []
            # Set keras callbacks
            self.cb_eStop = EarlyStopping(monitor='val_loss',patience=4,
                                          verbose=1,mode='auto')
            self.cb_check = []
            # Reset keras graph
            keras.backend.tf.reset_default_graph()
            keras.backend.clear_session()
            
            
            # Initialize or load config file
            if os.path.isfile(self.configFN):
                config = configparser.ConfigParser()
                try:
                    config.read(self.configFN)
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
                    
            self.disp_msg = 'Welcome to ItATMIS 2.0'
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
            
    def data_save(self):
        if self.saved:
            self.disp_msg = 'No new changes'
            return
        try:
            w = QtWidgets.QWidget()
            suggest = os.path.join(self.datadir,'ItATMISdata.h5')
            save_path = QtWidgets.QFileDialog.getSaveFileName(w,'Save Data',suggest,"hdf5 file (*.h5)")
            if len(save_path[0])==0:
                return
            try:
                self.disp_msg = 'Saving data...'
                # convert strings to ascii
                file_list_ascii = [n.encode("ascii", "ignore") for n in self.file_list]
                datadir_ascii = [self.datadir.encode("ascii", "ignore")]
                if not self.model_arch_path == []:
                    model_arch_path_ascii = [self.model_arch_path.encode("ascii","ignore")]
                    model_weights_path_ascii = [self.model_weights_path.encode("ascii","ignore")]
                if not self.AnnotationFile == []:
                    annotationfile_ascii = [self.AnnotationFile.encode("ascii","ignore")]
                dt = h5py.special_dtype(vlen=bytes)
                
                print(save_path[0])
                with h5py.File(save_path[0],'w') as hf:
                    hf.create_dataset("ItATMISfile",data=True,dtype=np.bool)
                    hf.create_dataset("images",data=self.images,dtype=np.float)
                    hf.create_dataset("segmask",data=self.segmask,dtype=np.int)
                    hf.create_dataset("WSseg",data=self.WSseg,dtype=np.int)
                    hf.create_dataset("inds",data=self.inds,dtype=np.int)
                    hf.create_dataset("spatres",data=self.spatres,dtype=np.float)
                    hf.create_dataset("niftiAff",data=self.niftiAff,dtype=np.float)
                    hf.create_dataset("numClasses",data=self.numClasses,dtype=np.int)
                    hf.create_dataset("file_list", (len(file_list_ascii),1),dt, file_list_ascii)
                    hf.create_dataset("mask_list",data=np.array(self.mask_list),dtype=bool)
                    hf.create_dataset("datadir", (len(datadir_ascii),1),dt,datadir_ascii)
                    if not self.model_arch_path == []:
                        hf.create_dataset("model_arch_path", (len(model_arch_path_ascii),1),dt,model_arch_path_ascii)
                        hf.create_dataset("model_weights_path", (len(model_weights_path_ascii),1),dt,model_weights_path_ascii)
                    if not self.AnnotationFile == []:
                        hf.create_dataset("annotation_file",(len(annotationfile_ascii),1),dt,annotationfile_ascii)
                    hf.create_dataset("FNind",data=self.FNind,dtype=np.int)
                    hf.create_dataset("targets",data=self.targets,dtype=np.float)
                    
                self.saved = True
                self.disp_msg = 'Data saved'
            except Exception as e:
                self.error_msg = 'Error saving data'
                print(e)
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        except Exception as e:
            print(e)
    
    def data_load(self):
        # check for saved
        if not self.saved:
            unsaved_msg = "You have unsaved data that will be discarded. Continue?"
            reply = QtWidgets.QMessageBox.question(self, 'Unsaved Data', 
                             unsaved_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        
        
        # get file path
        w = QtWidgets.QWidget()
        data_tup = QtWidgets.QFileDialog.getOpenFileName(w, 'Select data file',
                                                         self.datadir, 'hdf5 files (*.h5)')
        data_path = data_tup[0]
        if len(data_path) == 0:
            return
        self.disp_msg = 'Loading previous data...'
        # clear current data
        self.segmask = []
        self.segOutput = []
        # check if ItATMIS data
        with h5py.File(data_path,'r') as hf:
            keys = list(hf.keys())
            file_check = 'ItATMISfile' in keys
        if not file_check:
            self.error_msg = 'Not an ItATMIS file'
            return
        
        # load saved data
        try:
            with h5py.File(data_path,'r') as hf:
                self.images = np.array(hf.get('images'))
                self.volshape = self.images.shape
                self.segmask = np.array(hf.get('segmask'))
                self.WSseg = np.array(hf.get('WSseg'))
                self.niftiAff = np.array(hf.get('niftiAff'))
                self.spatres = np.array(hf.get('spatres'))
                if 'inds' in list(hf.keys()):
                    self.inds= np.array(hf.get('inds'))
                if 'numClasses' in list(hf.keys()):
                    self.numClasses = np.array(hf.get('numClasses'))
                    self.ui.spinClass.setMaximum(self.numClasses)
                file_list_temp = hf.get('file_list')
                self.file_list = [n[0].decode('utf-8') for n in file_list_temp]
                self.mask_list = list(hf.get('mask_list'))
                datadir_temp = hf.get('datadir')
                self.datadir = datadir_temp[0][0].decode('utf-8')
                if 'model_weights_path' in list(hf.keys()):
                    modelpath_temp = hf.get('model_weights_path')
                    self.model_weights_path = modelpath_temp[0][0].decode('utf-8')
                    modelpath_temp = hf.get('model_arch_path')
                    self.model_arch_path = modelpath_temp[0][0].decode('utf-8')
                if 'annotation_file' in list(hf.keys()):
                    annotationfile_temp = hf.get('annotation_file')
                    self.AnnotationFile = annotationfile_temp[0][0].decode('utf-8')
                self.FNind = np.array(hf.get('FNind'))
                print(self.FNind)
                self.targets = np.array(hf.get('targets'))

            if not self.model_weights_path == []:
                # load model
                self.disp_msg = 'Loading saved model...'
                keras.backend.clear_session()
                self.graph = keras.backend.tf.get_default_graph()
                with self.graph.as_default():
                    with open(self.model_arch_path,'r') as file:
                        json_string = json.load(file)
                    self.model = model_from_json(json_string)
                    self.model.load_weights(self.model_weights_path)
                    self.adopt = Adam()
                    self.model.compile(optimizer=self.adopt, loss=dice_multi_loss)
            
            # initialize display
            self.InitDisplay()
            # display file list
            self.UpdateFNlist()
            # update view
            # self.updateIms()
            # self.updateLines()                
            self.disp_msg = 'Data loaded'
            
        except Exception as e:
            self.error_msg = 'Error loading data'
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        
    def InitDisplay(self):
        try:
            # calculate aspect ratios
            asprat1 = self.spatres[1]/self.spatres[0]
            asprat2 = self.spatres[1]/self.spatres[2]
            asprat3 = self.spatres[0]/self.spatres[2]

            # add View Boxes to graphics view
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

            # fix aspect ratios
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
            imshape = np.array(self.images.shape,dtype=np.int16)
            midinds = np.round(imshape/2).astype(np.int16)
            self.inds = midinds
            self.volshape = imshape


            # Add initial images            
            self.img_item_ax.setImage(self.images[midinds[0],...])
            self.img_item_cor.setImage(self.images[:,midinds[1],:])
            self.img_item_sag.setImage(self.images[:,:,midinds[2]])

            # create empty segmask
            if self.segmask == []:
                segsz = self.images.shape
                self.segmask = np.zeros(segsz)
            
            # create empty display mask
            msksiz = np.r_[self.volshape,4]
            msk = np.zeros(msksiz,dtype=np.float)
            for cc in range(self.numClasses):
                colA = np.r_[self.GetColor(cc+1),self.alph]
                segmask_curclass = self.segmask==cc+1
                msk[segmask_curclass,:] = colA
            self.mask = msk

            self.msk_item_ax.setImage(self.mask[midinds[0],...])
            self.msk_item_cor.setImage(self.mask[:,midinds[1],...])
            self.msk_item_sag.setImage(self.mask[:,:,midinds[2],...])

            # buffer for view lines
            buff = 10
            self.buff = np.round(np.array(buff/self.spatres)).astype(np.int16)
            # axial view lines
            self.line_ax_cor = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_ax_cor.setData(x = np.array([0,midinds[2]-self.buff[2],midinds[2]+self.buff[2],imshape[2]]),
                                     y = np.array([midinds[1],midinds[1],midinds[1],midinds[1]]))
            self.vbox_ax.addItem(self.line_ax_cor)
            
            self.line_ax_sag = pg.PlotDataItem(pen='y',connect='pairs')
            self.line_ax_sag.setData(x = np.array([midinds[1],midinds[1],midinds[1],midinds[1]]),
                                     y = np.array([0,midinds[2]-self.buff[2],midinds[2]+self.buff[2],imshape[2]]))
            self.vbox_ax.addItem(self.line_ax_sag)

            # hide when not in use
            self.line_ax_cor.setAlpha(0,False)
            self.line_ax_sag.setAlpha(0,False)

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

            # adjust viewbox ranges
            self.vbox_ax.setRange(xRange=(0,imshape[2]),yRange=(0,imshape[1]),
                                  disableAutoRange=True,padding=0.)
            self.vbox_cor.setRange(xRange=(0,imshape[2]),yRange=(0,imshape[0]),
                                  disableAutoRange=True,padding=0.)
            self.vbox_sag.setRange(xRange=(0,imshape[1]),yRange=(0,imshape[0]),
                                  disableAutoRange=True,padding=0.)

            # calculate window/leveling multiplier
            self.WLmult = self.img_item_ax.getLevels()[1]/500
            
            # create starting brush            
            self.BrushMult = 1
            self.brush = self.my_brush_mask(self.ui.slideBrushSize.value()*self.BrushMult)
            self.rad = self.ui.slideBrushSize.value()
            
            # create brush cursor
            self.dot = pg.ScatterPlotItem(x=np.array([5.]),y=np.array([5.]),
                                    symbol='o',symbolSize=2*self.rad+1,symbolPen=(100,100,100,.5),
                                    brush=None,pxMode=False)
            self.vbox_ax.addItem(self.dot)
            pg.SignalProxy(self.ui.viewAxial.scene().sigMouseMoved, rateLimit=200, slot=self.mouseMoved)
            self.ui.viewAxial.scene().sigMouseMoved.connect(self.mouseMoved)

            self.dot.mouseClickEvent = self.testclick
            
            # Setup slider
            self.ui.slideAx.setMaximum(self.images.shape[0]-1)
            self.ui.slideAx.setValue(self.inds[0])
            
            # attach call backs
            self.ui.slideAx.valueChanged.connect(self.slide)
            self.ui.slideAlpha.valueChanged.connect(self.alphaSlide)
            self.ui.slideBrushSize.valueChanged.connect(self.brushSlide)

            self.ui.viewAxial.keyPressEvent = self.vbKeyPress
            self.ui.viewAxial.wheelEvent = self.scroll
            self.img_item_ax.mouseClickEvent = self.axClickEvent
            self.img_item_ax.mouseDragEvent = self.axDragEvent
            self.img_item_ax.hoverEvent = self.brushHover

            self.ui.viewCoronal.wheelEvent = self.corScroll
            self.img_item_cor.mouseClickEvent = self.corClickEvent
            self.img_item_cor.mouseDragEvent = self.corDragEvent

            self.ui.viewSag.wheelEvent = self.sagScroll
            self.img_item_sag.mouseClickEvent = self.sagClickEvent
            self.img_item_sag.mouseDragEvent = self.sagDragEvent
            
            # make full screen
            scrnsiz = QtWidgets.QDesktopWidget().screenGeometry()
            cntr = scrnsiz.center()
            width = 1100
            height = 720
            xv = np.round(cntr.x()-width/2)
            yv = np.round(cntr.y()-height/2)
            sizRect = QtCore.QRect(xv,yv,width,height)
            self.setGeometry(sizRect)
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))

    def updateIms(self):
        self.img_item_ax.setImage(self.images[self.inds[0],...],autoLevels=False)
        self.img_item_cor.setImage(self.images[:,self.inds[1],:],autoLevels=False)
        self.img_item_sag.setImage(self.images[:,:,self.inds[2]],autoLevels=False)
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
        
    def slide(self,ev):
        try:
            self.img_item_ax.setImage(self.images[ev,...],autoLevels=False)
            self.msk_item_ax.setImage(self.mask[ev,...])
            self.inds[0] = ev
            self.updateLines()
            
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def scroll(self,event):
        mods = QtWidgets.QApplication.keyboardModifiers()
        if mods == QtCore.Qt.ControlModifier:
            bump = -event.angleDelta().y()/120
            fac = 1+.1*bump
            pos = self.vbox_ax.mapToView(event.pos())
            z = self.inds[0]
            self.vbox_ax.scaleBy(s=fac,center=(pos.x(),pos.y()))
            self.vbox_cor.scaleBy(s=fac,center=(pos.x(),z))
            self.vbox_sag.scaleBy(s=fac,center=(pos.y(),z))
            event.accept()
        else:
            curval = self.ui.slideAx.value()
            newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                             self.volshape[0]-1))
            self.ui.slideAx.setValue(newval)

    def corScroll(self,ev):
        curval = self.inds[1]
        newval = np.int16(np.clip(curval+ev.angleDelta().y()/120,0,
                         self.volshape[1]-1))
        self.inds[1]= newval
        self.corUpdate(newval)
        self.updateLines()

    def corUpdate(self,value):
        self.img_item_cor.setImage(self.images[:,value,:],autoLevels=False)
        self.msk_item_cor.setImage(self.mask[:,self.inds[1],...])       
         
    def sagScroll(self,event):
        curval = self.inds[2]
        newval = np.int16(np.clip(curval+event.angleDelta().y()/120,0,
                         self.volshape[2]-1))
        self.inds[2] = newval
        self.sagUpdate(newval)
        self.updateLines()
            
    def sagUpdate(self,value):
        self.img_item_sag.setImage(self.images[:,:,value],autoLevels=False)
        self.msk_item_sag.setImage(self.mask[:,:,self.inds[2],...])

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
            if ev.isStart():
                self.line_ax_cor.setAlpha(1,False)
                self.line_ax_sag.setAlpha(1,False)
            elif ev.isFinish():
                self.line_ax_cor.setAlpha(0,False)
                self.line_ax_sag.setAlpha(0,False)
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.corMove(posx,posy)
            ev.accept()
            
        if ev.button()==4:
            if ev.isStart():
                self.prevLevel = np.array(self.img_item_cor.getLevels())
                self.startPos = np.array([ev.pos().x(),ev.pos().y()])
            else:
                self.levelEvent(ev)
            ev.accept()
                
    def corMove(self,x,y):
        aval = np.int16(np.clip(y,0,self.volshape[0]-1))
        self.inds[0] = aval
        self.ui.slideAx.setValue(aval)
        sval = np.int16(np.clip(x,0,self.volshape[2]-1))
        self.inds[2] = sval
        self.sagUpdate(sval)
        self.updateLines()
                                 
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
            if ev.isStart():
                self.line_ax_cor.setAlpha(1,False)
                self.line_ax_sag.setAlpha(1,False)
            elif ev.isFinish():
                self.line_ax_cor.setAlpha(0,False)
                self.line_ax_sag.setAlpha(0,False)
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.sagMove(posx,posy)
            ev.accept()
            
        if ev.button()==4:
            if ev.isStart():
                self.prevLevel = np.array(self.img_item_sag.getLevels())
                self.startPos = np.array([ev.pos().x(),ev.pos().y()])
            else:
                self.levelEvent(ev)
            ev.accept()
                
    def sagMove(self,x,y):
        aval = np.int16(np.clip(y,0,self.volshape[0]-1))
        self.inds[0] = aval
        self.ui.slideAx.setValue(aval)
        cval = np.int16(np.clip(x,0,self.volshape[1]-1))
        self.inds[1] = cval
        self.corUpdate(cval)
        self.updateLines()
    
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
            try: 
                newclass = int(ev.text())
                self.ui.spinClass.setValue(newclass)
            except ValueError:
                if ev.text() == ']':
                    curval = self.ui.slideBrushSize.value()
                    self.ui.slideBrushSize.setValue(curval+1)
                elif ev.text() == '[':
                    curval = self.ui.slideBrushSize.value()
                    self.ui.slideBrushSize.setValue(curval-1)
                elif ev.text() == '=':
                    curval = self.ui.slideAx.value()
                    newval = np.int16(np.clip(curval+1,0,
                                self.volshape[0]-1))
                    self.ui.slideAx.setValue(newval)
                elif ev.text() == '-':
                    curval = self.ui.slideAx.value()
                    newval = np.int16(np.clip(curval-1,0,
                                self.volshape[0]-1))
                    self.ui.slideAx.setValue(newval)
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
        self.updateIms()

    def mouseMoved(self,pos):
        if self.img_item_ax.sceneBoundingRect().contains(pos):
            mousePoint = self.vbox_ax.mapSceneToView(pos)
            self.dot.setData(x=np.array([mousePoint.x()]),y=np.array([mousePoint.y()]))

    def brushHover(self,ev):
        if ev.isEnter():
            self.dot.setData(symbolPen=(100,100,100,0.5))
        elif ev.isExit():
            self.dot.setData(symbolPen=(100,100,100,0))

    def testclick(self,ev):
        # Leave this here!
        ev.ignore()

    def axClickEvent(self,ev):
        if ev.button()==1 or ev.button()==2:
            # update undo list
            self.ui.actionUndo.setEnabled(True)
            self.segmask_undos.append(np.copy(self.segmask))
            if len(self.segmask_undos)>self.max_undo:
                _ = self.segmask_undos.pop(0)
            self.curclass = self.ui.spinClass.value()
            self.curmask = self.segmask[self.inds[0]]==self.curclass
            self.curws = self.WSseg[self.inds[0]]
            posx = ev.pos().x()
            posy = ev.pos().y()
            self.bt = ev.button()
            self.axMove(posx,posy)
            fmask = binary_fill_holes(self.curmask)
            colA = np.r_[self.GetColor(self.curclass),self.alph]
            self.mask[self.inds[0],self.segmask[self.inds[0]]==self.curclass,:] = 0
            self.mask[self.inds[0],fmask,:] = colA
            self.segmask[self.inds[0],self.segmask[self.inds[0]]==self.curclass] = 0
            self.segmask[self.inds[0],fmask] = self.curclass
            self.updateIms()
            self.saved = False
            ev.accept()
        else:
            ev.ignore()

    def axDragEvent(self,ev):
        if ev.button()==1 or ev.button()==2:
            if ev.isStart():
                # update undo list
                self.ui.actionUndo.setEnabled(True)
                self.segmask_undos.append(np.copy(self.segmask))
                if len(self.segmask_undos)>self.max_undo:
                    _ = self.segmask_undos.pop(0)
                self.curclass = self.ui.spinClass.value()
                self.curmask = self.segmask[self.inds[0]] == self.curclass
                self.curws = self.WSseg[self.inds[0]]
                self.prev_mask = np.copy(self.curmask)
                self.bt = ev.button()
                posx = ev.pos().x()
                posy = ev.pos().y()
                self.axMove(posx,posy)
            elif ev.isFinish():
                fmask = binary_fill_holes(self.curmask)
                colA = np.r_[self.GetColor(self.curclass),self.alph]
                self.mask[self.inds[0],self.segmask[self.inds[0]]==self.curclass] = 0
                self.mask[self.inds[0],fmask,:] = colA
                self.segmask[self.inds[0],self.segmask[self.inds[0]]==self.curclass] = 0
                self.segmask[self.inds[0],fmask] = self.curclass
                self.updateIms()
                self.saved = False
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
                self.levelEvent(ev)
            ev.accept()

    def axMove(self,x,y):
        cval = np.int16(np.clip(y,0,self.volshape[1]-1))
        self.inds[1]= cval
        self.corUpdate(cval)
        sval = np.int16(np.clip(x,0,self.volshape[2]-1))
        self.inds[2]= sval
        self.sagUpdate(sval)
        self.updateLines()
        if self.qs_on:
            self.quickDraw(x,y)
        else:
            self.draw(x,y)

    def draw(self,x,y):
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
                reg = np.maximum(pbrush,reg)
                cmask[lby:uby,lbx:ubx] = reg          
            else:
                self.mask[self.inds[0],cmask,:] = 0
                reg = np.minimum(1-np.minimum(pbrush,reg),reg)
                cmask[lby:uby,lbx:ubx] = reg                
                
        self.curmask = cmask
        colA = np.r_[self.GetColor(self.curclass),self.alph]
        self.mask[self.inds[0],cmask,:] = colA
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])
    
    def quickDraw(self,x,y):
        brush = self.brush
        cmask = self.curmask
        pmask = np.zeros_like(cmask,dtype=np.bool)
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
            pmask[lby:uby,lbx:ubx] = pbrush
            wsvals = np.unique(self.curws[pmask==1])
            recon = np.zeros(self.curws.shape,dtype=np.bool)
            for val in wsvals:
                recon+=(self.curws==val)
            qmask = reconstruction(pmask,recon)

            if bt == 1:
                cmask = np.maximum(qmask,cmask).astype(np.int)  
            else:
                self.mask[self.inds[0],cmask,:] = 0
                cmask = np.minimum(1-np.minimum(qmask,cmask),cmask).astype(np.int)
        cmask = cmask.astype(np.bool)
        self.curmask = cmask
        colA = np.r_[self.GetColor(self.curclass),self.alph]
        self.mask[self.inds[0],cmask,:] = colA
        self.msk_item_ax.setImage(self.mask[self.inds[0],...])

    def undo(self):
        try:
            if len(self.segmask_undos)>0:
                self.segmask = self.segmask_undos.pop(-1)
                self.mask = 0*self.mask
                for cc in range(self.numClasses):
                    colA = np.r_[self.GetColor(cc+1),self.alph]
                    segmask_curclass = self.segmask==cc+1
                    self.mask[segmask_curclass,:] = colA
                self.updateIms()
            if len(self.segmask_undos) == 0:
                self.ui.actionUndo.setEnabled(False)
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            
    def clearMask(self):
        # check to quit
        quit_msg = "Clear slice or entire 3D mask?"
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setWindowTitle('Clear Mask')
        msg.setText(quit_msg)
        a_slice = msg.addButton(
            'Slice', QtWidgets.QMessageBox.AcceptRole)
        a_full = msg.addButton(
            'Full 3D', QtWidgets.QMessageBox.AcceptRole)
        a_cancel = msg.addButton(
            'Cancel', QtWidgets.QMessageBox.RejectRole)
        msg.setDefaultButton(a_slice)
        msg.setEscapeButton(a_cancel)
        msg.exec_()
        # msg.deleteLater()
        print(msg.clickedButton())
        if msg.clickedButton() is a_slice:
            print('Clearing slice')
            self.segmask[self.inds[0]] = 0
            self.mask[self.inds[0]] = 0
            self.updateIms()
            self.saved = False
            self.disp_msg = 'Current slice cleared'
        elif msg.clickedButton() is a_full:
            print('Clearing entire mask')
            self.segmask = np.zeros_like(self.segmask)
            self.mask = 0*self.mask
            self.updateIms()
            self.saved = False
            self.disp_msg = 'Full mask cleared'
        else:
            print('Canceled')
            msg.close()
                
    def quick_select(self,ev):
        self.qs_on = ev

    def levelEvent(self,ev):
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
        
    def DataSelect(self):
        self.DataSelectW = DataSelect(self)
        self.DataSelectW.show()

    def UpdateFNlist(self):
        # make bold font
        bFont = QFont()
        bFont.setBold(True)
        nFont = QFont()
        # add to list view
        for fname in self.file_list:
            _,fn = os.path.split(fname)
            item = QtWidgets.QListWidgetItem(fn)
            item.setFont(bFont)
            self.ui.listFiles.addItem(item)
        # set current selection
        self.ui.listFiles.setCurrentRow(self.FNind)
        # Update fonts according to mask list
        for ii in range(len(self.mask_list)):
            if self.mask_list[ii]:
                self.ui.listFiles.item(ii).setFont(nFont)
        
    
    def FileListSelect(self,ev):
        newFNind = self.ui.listFiles.row(ev)
        print(newFNind)
        self.ChangeSubject(newFNind)

    def FileListClick(self,ev):
        self.ui.listFiles.setCurrentRow(self.FNind)

    def SaveCurrentMask(self):
         # Save current mask to nifti
        # only if mask has annotations
        if not np.sum(self.segmask)==0:
            # create nifti image with same affine
            data = np.swapaxes(np.rollaxis(self.segmask,2,0),1,2).astype(np.int)
            img = nib.Nifti1Image(data, self.niftiAff)
            # generate mask file name
            curFile = self.file_list[self.FNind]
            fdir,fFN = os.path.split(curFile)
            maskdir = os.path.join(fdir,'ItATMISmasks')
            FN,ext = os.path.splitext(fFN)
            maskFN = FN + '_mask' + ext
            maskPath = os.path.join(maskdir,maskFN)
            # check if mask directory has been made already
            if not os.path.exists(maskdir):
                os.mkdir(maskdir)
            # save to nifti
            with contextlib.suppress(FileNotFoundError):
                os.remove(maskPath)
            nib.save(img,maskPath)
            # update list of mask files
            self.mask_list[self.FNind] = True
            nFont = QFont()
            self.ui.listFiles.item(self.FNind).setFont(nFont)

    def ChangeSubject(self,ind):
        try:
            # save current subject
            self.SaveCurrentMask()
            # update file index
            self.FNind = ind
            # load new subject
            self.ImportImages(False)
        except Exception as e:
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        
    def ImportImages(self,segAfter):
        self.disp_msg = "Importing subject {}...".format(self.FNind+1)
        curFN = self.file_list[self.FNind]
        if not self.training:
            self.ui.progBar.setVisible(True)
            self.ui.progBar.setRange(0,0)
        self.imp_thread = NiftiImportThread(curFN,segAfter,self.numClasses)
        self.imp_thread.finished.connect(self.imp_finish_imp)
        self.imp_thread.data_sig.connect(self.gotData)
        self.imp_thread.WS_sig.connect(self.gotWSseg)
        self.imp_thread.images_sig.connect(self.gotImages)
        self.imp_thread.errorsig.connect(self.impError)
        self.imp_thread.segmask_error.connect(self.impSegError)
        self.imp_thread.start()
    
    def imp_finish_imp(self):
        return
    
    def gotWSseg(self,WSseg):
        self.WSseg = WSseg

    def gotData(self,aff,spatres):
        self.niftiAff = aff
        self.spatres = spatres

    def gotImages(self,images,segmask,segAfter):
        self.images = images
        self.volshape = images.shape
        self.segmask = segmask
        self.disp_msg = 'Images Imported'
        self.saved = False
        if segAfter:
            self.disp_msg = 'Segmenting images...'
            self.EvalCurrentSubject()
        else:
            if not self.training:
                self.ui.progBar.setRange(0,1)
                self.ui.progBar.setTextVisible(False)
                self.ui.progBar.setVisible(False)
            self.ui.viewAxial.setEnabled(True)
            self.InitDisplay()
    
    def impError(self):
        self.error_msg = 'Error importing images'
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        self.ui.viewAxial.setEnabled(True)

    def impSegError(self,msg):
        self.error_msg = msg
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        self.ui.viewAxial.setEnabled(True)

    def seg_finish(self):
        self.disp_msg = 'Segmentation complete'
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        self.ui.viewAxial.setEnabled(True)
        self.mask = 0*self.mask
        for cc in range(self.numClasses):
            colA = np.r_[self.GetColor(cc+1),self.alph]
            segmask_curclass = self.segmask==cc+1
            self.mask[segmask_curclass,:] = colA
        self.updateIms()
        
    def seg_gotmask(self,mask):
        self.segmask = mask
        
    def seg_error(self,error):
        self.error_msg = error
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        self.ui.viewAxial.setEnabled(True)
    
    def Train(self):
        try:
            # save current mask
            self.SaveCurrentMask()

            # Initialize progress bar
            self.ui.progBar.setVisible(True)
            self.ui.progBar.setRange(0,0)
            # block other progress bar updates
            self.training = True
            
            if self.model_arch_path == []:
                timestr = time.strftime("%Y%m%d%H")
                self.model_weights_path = os.path.join(self.datadir,'ItATMISmodel_{}.h5'.format(timestr))
                self.model_arch_path = os.path.join(self.datadir,'ItATMISmodel_{}.txt'.format(timestr))

            # Generate model if not existant
            if self.model == []:                
                print('Generating model...')
                self.graph = keras.backend.tf.get_default_graph()
                with self.graph.as_default():
                    # create optimizer
                    self.adopt = Adam()
                    # create model
                    self.model = BlockModel(self.images.shape,num_out_channels=self.numClasses+1)
                    # save architecture
                    json_model = self.model.to_json()
                    with open(self.model_arch_path,'w') as file:
                        json.dump(json_model,file)

                self.model.compile(optimizer=self.adopt, loss=dice_multi_loss)
                
            # Set checkpoint callback if not already set
            if self.cb_check == []:
                print('Making callback...')
                self.cb_check = ModelCheckpoint(self.model_weights_path,monitor='val_loss',
                                           verbose=0,save_best_only=True,
                                           save_weights_only=True,
                                           mode='auto',period=1)
            
            print('Creating training thread...')
            # Start Training Thread
            self.train_thread = TrainThread(self.graph, self.model,
                                        self.file_list, self.FNind, self.mask_list,
                                        self.cb_eStop,self.cb_check, self.model_weights_path,
                                        self.numClasses)
            self.train_thread.message_sig.connect(self.trainGotMessage)
            self.train_thread.model_sig.connect(self.trainGotModel)
            self.train_thread.finished.connect(self.trainFinished)
            self.train_thread.error_sig.connect(self.trainError)
            self.train_thread.train_sig.connect(self.trainStart)
            self.train_thread.batch_sig.connect(self.trainBatch)
            self.train_thread.start()
        except Exception as e:
            self.error_msg = str(e)
            self.ui.progBar.setRange(0,1)
            self.ui.progBar.setTextVisible(False)
            self.ui.progBar.setVisible(False)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.ui.viewAxial.setEnabled(True)
            
    def trainGotMessage(self, msg):
        self.disp_msg = msg
    
    def trainGotModel(self, model,graph):
        self.model = model
        self.graph = graph
        
    def trainFinished(self):
        self.training = False
        self.EvalCurrentSubject()
        
    def trainError(self, msg):
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.viewAxial.setEnabled(True)
        self.error_msg = msg

    def trainStart(self):
        print('Training Started')
        self.ui.progBar.setRange(0,100)
        self.ui.progBar.setTextVisible(True)
        
    def trainBatch(self,num):
        self.ui.progBar.setValue(num)       
        
    def EvalCurrentSubject(self):
        self.disp_msg = 'Evaluating current subject'
        self.ui.viewAxial.setEnabled(False)
        self.seg_thread = SegThread(self.graph,self.images,self.segmask,self.model)
        self.seg_thread.finished.connect(self.seg_finish)
        self.seg_thread.segmask_sig.connect(self.seg_gotmask)
        self.seg_thread.error_sig.connect(self.seg_error)
        self.seg_thread.start()
                
    def resetView(self,ev):
        self.vbox_ax.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[1]))
        self.vbox_cor.setRange(xRange=(0,self.volshape[2]),yRange=(0,self.volshape[0]))
        self.vbox_sag.setRange(xRange=(0,self.volshape[1]),yRange=(0,self.volshape[0]))
        self.img_item_ax.setImage(self.images[self.inds[0],...],autoLevels=True)
        self.img_item_cor.setImage(self.images[:,self.inds[1],:],autoLevels=True)
        self.img_item_sag.setImage(self.images[:,:,self.inds[2]],autoLevels=True)
        
    #%%
    def GetColor(self,ind):
        colors = np.array([
            [1,1,0],
            [0,1,0],
            [1,0,0],
            [0,0,1],
            [0,1,1],
            [1,0,1]
            ])
        return colors[ind-1]

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
    
    @pyqtProperty(bool)
    def saved(self):
        return self._saved
    @saved.setter
    def saved(self, value):
        self._saved = value
        self.ui.action_Save_Data.setEnabled(not value)
        QtWidgets.qApp.processEvents()
        
    def closeEvent(self, ev):
        # check to quit
        if not self.saved:
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
        
        # kill threads
        try:
            self.train_thread.quit()
        except Exception:
            pass
        # exit
        del self

#%%
def BlockModel(in_shape,filt_num=16,numBlocks=4,num_out_channels=2):
    input_shape = in_shape[1:]+(1,)
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    #calculate appropriate cropping
    mod = np.mod(input_shape[0:2],2**numBlocks)
    padamt = mod+2
    # calculate size reduction
    startsize = np.max(input_shape[0:2]-padamt)
    minsize = (startsize-np.sum(2**np.arange(1,numBlocks+1)))/2**numBlocks
    if minsize<4:
        numBlocks=3
    
    crop = Cropping2D(cropping=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_input)

    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(crop)
    lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(crop)
    lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(crop)
    lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # rest of contracting blocks
    for rr in range(2,numBlocks+1):
        lay_conv1 = Conv2D(filt_num*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(filt_num*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filt_num*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(filt_num*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # last expanding block
    dd=numBlocks
    lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(filt_num*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)    
    lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # rest of expanding blocks
    expnums = list(range(1,numBlocks))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(filt_num*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(filt_num*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(filt_num*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(filt_num*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(filt_num*dd, (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(filt_num*dd, (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,padamt[0]), (0,padamt[1])), data_format=None)(lay_act)
        
    # segmenter
    lay_out = Conv2D(num_out_channels,(1,1), activation='sigmoid',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)

#%% Dice Coefficient
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
#%% Dice loss
def dice_multi_loss(y_true, y_pred):
    numLabels = K.int_shape(y_pred)[-1]
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[...,index], y_pred[...,index])
    return dice

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
            full_path,_ = QtWidgets.QFileDialog.getOpenFileName(
                                            w, 'Select image file',
                                            self.parent.datadir,
                                            filters,selected_filter)
            
            if len(full_path)==0:
                return
            filedir,FN = os.path.split(full_path)
            FN,_ = os.path.splitext(FN)
            
            # generate file names
            self.parent.datadir = filedir
            imp_list = []  # create an empty list
            for item in os.listdir(filedir):
                if item.endswith('.nii'):
                    imp_list.append(os.path.join(filedir,item))
            imp_list = natsorted(imp_list)
            
            FNs = []
            # add to list view
            for fname in imp_list:
                FNs.append(fname)
                _,fn = os.path.split(fname)
                item = QtWidgets.QListWidgetItem(fn)
                self.ui.list_files.addItem(item)
                
            self.FNs = FNs
            self.dirName = filedir
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
                         FNs.append(os.path.join(self.dirName,self.ui.list_files.item(index).text()))
                    self.FNs = FNs
        except Exception as e:
            print(e)
            
    def setSelect(self):
        if not len(self.FNs) == 0:
            self.parent.numClasses = self.ui.spinBox.value()
            print('Number of classes selected is',self.parent.numClasses)
            self.parent.ui.spinClass.setMaximum(self.parent.numClasses)
            self.parent.file_list = self.FNs
            self.parent.mask_list = [False]*len(self.FNs)
            self.parent.UpdateFNlist()
            self.parent.disp_msg = 'Files selected'
            self.parent.saved = False
            self.hide()
            self.parent.ImportImages(False)
            self.destroy()


#%%
class NiftiImportThread(QThread):
    images_sig = pyqtSignal(np.ndarray,np.ndarray,bool)
    data_sig = pyqtSignal(np.ndarray,np.ndarray)
    WS_sig = pyqtSignal(np.ndarray)
    errorsig = pyqtSignal()
    segmask_error = pyqtSignal(str)
    def __init__(self,FN,segAfter,numClasses):
        QThread.__init__(self)
        self.FN = FN
        self.segAfter = segAfter
        self.numClasses = numClasses

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
            # import nifti
            nft = nib.load(self.FN)
            
            # adjust orientation
            canon_nft = nib.as_closest_canonical(nft)
            aff = canon_nft.affine
            # Load spacing values (in mm)
            spatres = np.array(canon_nft.header.get_zooms())
            # send to main app
            self.data_sig.emit(aff,spatres)
            
            # format and normalize images
            ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
            for im in ims:
                im -= np.min(im)
                im /= np.max(im)
            # create pre-segmentation for quick select
            WSseg = np.zeros_like(ims)
            for ss in range(WSseg.shape[0]):
                im = ims[ss,...]
                imgrad = sobel(im)
                imgrad[imgrad<.01] = 0
                WSseg[ss,...] = watershed(imgrad, markers=800, compactness=0.001)

            # look for mask and load, if there is one
            fdir,fFN = os.path.split(self.FN)
            maskdir = os.path.join(fdir,'ItATMISmasks')
            FN,ext = os.path.splitext(fFN)
            maskFN = FN + '_mask' + ext
            maskPath = os.path.join(maskdir,maskFN)
            if os.path.exists(maskPath):
                try:
                    segnft = nib.as_closest_canonical(nib.load(maskPath))
                    segmask = np.swapaxes(np.rollaxis(segnft.get_data(),2,0),1,2).astype(np.int)
                except Exception as e:
                    print(e)
                    self.segmask_error.emit('Unable to load mask')
                    segmask = np.zeros_like(ims,dtype=np.int)
            else:
                segmask = np.zeros_like(ims,dtype=np.int)
            
            self.WS_sig.emit(WSseg)
            self.images_sig.emit(ims,segmask,self.segAfter)
                
        except Exception as e:
            self.errorsig.emit()
            print(e)
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            self.quit()

#%%
class TrainThread(QThread):
    error_sig = pyqtSignal(str)
    message_sig = pyqtSignal(str)
    model_sig = pyqtSignal(keras.engine.training.Model,keras.backend.tf.Graph)
    train_sig = pyqtSignal()
    batch_sig = pyqtSignal(int)
    
    def __init__(self,graph,model,file_list,FNind,mask_list,
                 CBstop,CBcheck,model_weights_path,
                 numClasses):
        QThread.__init__(self)
        self.graph = graph
        self.file_list = file_list
        self.FNind = FNind
        self.model = model
        self.mask_list = mask_list
        self.CBs = [CBstop,CBcheck]
        self.model_weights_path = model_weights_path
        self.numClasses = numClasses
        
    def __del__(self):
        self.terminate()

    def GenerateMaskPath(self,FNind):
        # generate mask file name
        curFile = self.file_list[FNind]
        fdir,fFN = os.path.split(curFile)
        maskdir = os.path.join(fdir,'ItATMISmasks')
        FN,ext = os.path.splitext(fFN)
        maskFN = FN + '_mask' + ext
        maskPath = os.path.join(maskdir,maskFN)
        return maskPath

    def run(self):
        try:
            # load current images
            self.message_sig.emit('Loading inputs...')
            # get list of files with masks
            print(self.mask_list)
            file_inds = np.arange(len(self.mask_list))[self.mask_list]
            print(file_inds)
            # import first nifti and mask
            curind = file_inds[0]
            nft = nib.load(self.file_list[curind])
            masknft = nib.load(self.GenerateMaskPath(curind))
            # adjust orientation
            canon_nft = nib.as_closest_canonical(nft)
            canon_masknft = nib.as_closest_canonical(masknft)
            ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
            mask = np.swapaxes(np.rollaxis(canon_masknft.get_data(),2,0),1,2)
            # normalize
            for im in ims:
                im -= np.min(im)
                im /= np.max(im)
            # add axis
            inputs = ims[...,np.newaxis]
            targets = to_categorical(mask,self.numClasses+1)
            # import rest of niftis, if more than 1 subject
            if len(file_inds) > 1:
                for ss in file_inds[1:]:
                    # load next subject
                    nft = nib.load(self.file_list[ss])
                    masknft = nib.load(self.GenerateMaskPath(curind))
                    # adjust orientation
                    canon_nft = nib.as_closest_canonical(nft)
                    canon_masknft = nib.as_closest_canonical(masknft)
                    ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
                    mask = np.swapaxes(np.rollaxis(canon_masknft.get_data(),2,0),1,2)
                    for im in ims:
                        im -= np.min(im)
                        im /= np.max(im)
                    # add axis and concatenate to target array
                    inputs = np.concatenate((inputs,ims[...,np.newaxis]),axis=0)
                    newmask = to_categorical(mask,self.numClasses+1)
                    targets = np.concatenate((targets,newmask),axis=0)
            
            # check for equal sizes
            if targets.shape[:-1] != inputs.shape[:-1]:
                print('Targets shape is:', targets.shape)
                print('Inputs shape is:', inputs.shape)
                raise ValueError('Input and target shape do not match')
                
            # split off validation data
            self.message_sig.emit('Splitting data...')
            numIm = inputs.shape[0]
            val_inds = np.random.choice(np.arange(numIm),
                                        np.round(.2*numIm).astype(np.int),
                                        replace=False)
            valX = np.take(inputs,val_inds,axis=0)
            valY = np.take(targets,val_inds, axis=0)
            trainX = np.delete(inputs, val_inds, axis=0)
            trainY = np.delete(targets, val_inds, axis=0)
            
            # save training data for testing
            TrainingFile = 'TrainingData.h5'
            with h5py.File(TrainingFile,'w') as hf:
                hf.create_dataset("trainX", data=trainX,dtype='f')
                hf.create_dataset("trainY",data=trainY,dtype='f')
                hf.create_dataset("valX",data=valX,dtype='f')
                hf.create_dataset("valY",data=valY,dtype='f')
            
            
            # setup image data generator
            datagen1 = ImageDataGenerator(
                rotation_range=15,
                shear_range=0.5,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')
            datagen2 = ImageDataGenerator(
                rotation_range=15,
                shear_range=0.5,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')
     
            # Provide the same seed and keyword arguments to the fit and flow methods
            seed = 1
            datagen1.fit(trainX, seed=seed)
            datagen2.fit(trainY, seed=seed)
            batchsize = 16
            datagen = zip( datagen1.flow( trainX, None, batchsize, seed=seed), datagen2.flow( trainY, None, batchsize, seed=seed) )
            
            # calculate number of epochs and batches
            # numEp = np.maximum(40,np.minimum(np.int(10*(self.FNind+1)),100))
            numEp = 30
            steps = np.minimum(np.int(trainX.shape[0]/batchsize*8),1000)
            numSteps = steps*numEp
            
            # Make progress callback
            progCB = ProgressCallback()
            progCB.thread = self
            progCB.progress = 0
            progCB.batchpercent = 100/(numSteps)
            self.CBs = self.CBs + [progCB]
            
            self.message_sig.emit('Training model...')
            with self.graph.as_default():
                self.model.fit_generator(datagen,
                                         steps_per_epoch=steps,
                                         epochs=numEp,
                                         callbacks=self.CBs,
                                         validation_data=(valX,valY))
            
            self.batch_sig.emit(100)    
            
            # keras.backend.clear_session()
            # keras.backend.tf.reset_default_graph()
                
            self.message_sig.emit('Training Complete.')
            # self.message_sig.emit('Loading best model...')
            # graph = keras.backend.tf.get_default_graph()
            # with self.graph.as_default():
                # self.model = load_model(self.model_path,custom_objects={'dice_loss':dice_loss})

            with self.graph.as_default():
                self.model.load_weights(self.model_weights_path)
            
            # self.message_sig.emit('Evaluating on validation data...')
            # score = self.model.evaluate(valX,valY,verbose=0)
            # self.message_sig.emit('Dice Score: {:.2f}'.format(1-score))
            
            self.model_sig.emit(self.model, self.graph)
            
        except Exception as e:
            self.error_sig.emit(str(e))
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
            self.quit()
#%%
class ProgressCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.thread.train_sig.emit()
        return

    def on_epoch_end(self, epoch, logs={}):
        self.progress = self.progress + self.batchpercent
        self.thread.batch_sig.emit(int(self.progress))
        return
    
    def on_batch_end(self, batch, logs={}):
        self.progress = self.progress + self.batchpercent
        self.thread.batch_sig.emit(int(self.progress))
        return
#%%
class SegThread(QThread):
    segmask_sig = pyqtSignal(np.ndarray)
    error_sig = pyqtSignal(str)
    
    def __init__(self,graph,ims,mask,model):
        QThread.__init__(self)
        self.graph = graph
        self.ims = ims
        self.mask = mask
        self.model = model
        
    def __del__(self):
        self.wait()

    def run(self):
        try:
            # determine which slices have been segmented already
            slice_inds = np.where(self.mask.any(axis=(1,2)))[0]
            inputs = self.ims[...,np.newaxis]
            
            with self.graph.as_default():
                output = self.model.predict(inputs,batch_size=16,verbose=0)
                
            mask = np.argmax(output,axis=-1)
            mask[slice_inds,...] = self.mask[slice_inds,...]
                    
            self.segmask_sig.emit(mask)
        except Exception as e:
            self.error_sig.emit(str(e))
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
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
    app.exec_()