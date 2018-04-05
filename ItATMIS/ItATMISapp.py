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
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
import configparser
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
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model

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
            self.ui.pb_Train.clicked.connect(self.Train)
            
            # initialize display message
            self.disp_msg = 'Initializing...'
            
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
            # HDF5 file of annotations
            self.AnnotationFile = []
            # deep learning model
            self.model = []
            self.model_path = []
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
            # keras graph
            self.graph = []
            # Set keras callbacks
            self.cb_eStop = EarlyStopping(monitor='val_loss',patience=5,
                                          verbose=0,mode='auto')
            self.cb_check = []
            # Set keras optimizer
            keras.backend.tf.reset_default_graph()
            keras.backend.clear_session()
            self.adopt = Adam()
            
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
        print('Calculating')
        
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
        
    def DataSelect(self):
        self.DataSelectW = DataSelect(self)
        self.DataSelectW.show()
        
    def ImportImages(self,segAfter):
        self.disp_msg = "Importing subject {}...".format(self.FNind+1)
        curFN = self.file_list[self.FNind]
        self.ui.progBar.setVisible(True)
        self.ui.progBar.setRange(0,0)
        # Set cursor to wait
        QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
        self.imp_thread = NiftiImportThread(curFN,segAfter)
        self.imp_thread.finished.connect(self.imp_finish_imp)
        self.imp_thread.images_sig.connect(self.gotImages)
        self.imp_thread.errorsig.connect(self.impError)
        self.imp_thread.start()
    
    def imp_finish_imp(self):
        pass
    
    def gotImages(self,images,segAfter):
        self.images = images
        self.volshape = images.shape
        self.disp_msg = 'Images Imported'
        self.saved = False
        if segAfter:
            self.disp_msg = 'Segmenting images...'
            self.seg_thread = SegThread(self.graph,self.images,self.model)
            self.seg_thread.finished.connect(self.seg_finish)
            self.seg_thread.segmask_sig.connect(self.seg_gotmask)
            self.seg_thread.error_sig.connect(self.seg_error)
            self.seg_thread.start()
        else:
            self.ui.progBar.setRange(0,1)
            self.ui.progBar.setTextVisible(False)
            self.ui.progBar.setVisible(False)
            time.sleep(.1)
            QtWidgets.QApplication.restoreOverrideCursor()
            self.ui.viewAxial.setEnabled(True)
            self.InitDisplay()
    
    def impError(self):
        self.error_msg = 'Error importing images'
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.viewAxial.setEnabled(True)
        
    def seg_finish(self):
        self.disp_msg = 'Segmentation complete'
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.viewAxial.setEnabled(True)
        self.img_item.setImage(self.images[self.ind,...],autoLevels=False)
        self.msk_item.setImage(self.mask[self.ind,...])
        self.mask[...,3] = self.alph*self.segmask
        
    
    def seg_gotmask(self,mask):
        self.segmask = mask
        
    def seg_error(self,error):
        self.error_msg = error
        self.ui.progBar.setRange(0,1)
        self.ui.progBar.setTextVisible(False)
        self.ui.progBar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.ui.viewAxial.setEnabled(True)
                
    def PrepareTargets(self):
        if len(self.segmask) != 0:
            self.targets = self.segmask[...,np.newaxis]
    
    def Train(self):
        try:
            # Initialize progress bar
            self.ui.progBar.setVisible(True)
            self.ui.progBar.setRange(0,0)
            # Set cursor to wait
            QtWidgets.QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))
            # Disable annotation window
            self.ui.viewAxial.setEnabled(False)
            
            # Prepare current inputs and targets
            self.disp_msg = 'Preparing data...'
            self.PrepareTargets()
            # Save or update annotations
            if len(self.AnnotationFile) ==0:
                # make original annotation file
                AnnotationFile = os.path.join(self.datadir,'Annotations.h5')
                with h5py.File(AnnotationFile,'w') as hf:
                    hf.create_dataset("targets", data=self.targets,dtype='f')
                    self.AnnotationFile = AnnotationFile
            else:
                # load current annotation file and append
                with h5py.File(self.AnnotationFile,'r') as f:
                    old = np.array(f.get('targets'))
                self.targets = np.concatenate((old,self.targets),axis=0)
                with h5py.File(self.AnnotationFile,'w') as hf:
                    hf.create_dataset("targets", data=self.targets,dtype='f')
            # Generate model if not existant
            if self.model == []:                
                self.disp_msg = 'Generating model...'
                self.graph = keras.backend.tf.get_default_graph()
                with self.graph.as_default():
                    self.model = BlockModel(self.images.shape)
                self.model.compile(optimizer=self.adopt, loss=dice_loss)
                
            # Set checkpoint callback if not already set
            if self.cb_check == []:
                print('Making callback...')
                self.model_path = os.path.join(self.datadir,'ItATMISmodel.h5')
                self.cb_check = ModelCheckpoint(self.model_path,monitor='val_loss',
                                           verbose=0,save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',period=1)
            
            print('Creating training thread...')
            # Start Training Thread
            self.train_thread = TrainThread(self.graph, self.model,
                                        self.file_list, self.FNind, self.targets,
                                        self.cb_eStop,self.cb_check, self.model_path)
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
        self.EvalNextSubject()
        
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
        
        
        
    def EvalNextSubject(self):
        self.FNind = self.FNind + 1
        self.ImportImages(True)
                
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
        
        # kill threads
        self.train_thread.quit()
        # exit
        del self

#%%
def BlockModel(in_shape,filt_num=16,numBlocks=4):
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
    lay_out = Conv2D(1,(1,1), activation='sigmoid',name='output_layer')(lay_pad)
    
    return Model(lay_input,lay_out)
#%% Dice Coefficient Loss
def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    return 1-dice

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
        self.parent.ImportImages(False)
        self.destroy()


#%%
class NiftiImportThread(QThread):
    images_sig = pyqtSignal(np.ndarray,bool)
    errorsig = pyqtSignal()
    def __init__(self,FN,segAfter):
        QThread.__init__(self)
        self.FN = FN
        self.segAfter = segAfter

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
            for im in ims:
                im -= np.min(im)
                im /= np.max(im)
            ims_send = self.noise_elim(ims)
            self.images_sig.emit(ims_send,self.segAfter)
                
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
    
    def __init__(self,graph,model,file_list,FNind,targets,
                 CBstop,CBcheck,model_path):
        QThread.__init__(self)
        self.graph = graph
        self.file_list = file_list
        self.FNind = FNind
        self.model = model
        self.targets = targets
        self.CBs = [CBstop,CBcheck]
        self.model_path = model_path
        
    def __del__(self):
        self.terminate()

    def run(self):
        try:
            # load current images
            self.message_sig.emit('Loading inputs...')
            # import first nifti
            nft = nib.load(self.file_list[0])
            # adjust orientation
            canon_nft = nib.as_closest_canonical(nft)
            ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
            # add axis
            inputs = ims[...,np.newaxis]
            # import rest of niftis, if more than 1 subject
            if self.FNind > 0:
                for ss in range(1,self.FNind+1):
                    # load next subject
                    nft = nib.load(self.file_list[ss])
                    # adjust orientation
                    canon_nft = nib.as_closest_canonical(nft)
                    ims = np.swapaxes(np.rollaxis(canon_nft.get_data(),2,0),1,2)
                    # add axis and concatenate to target array
                    inputs = np.concatenate((inputs,ims[...,np.newaxis]),axis=0)
            
            # check for equal sizes
            if self.targets.shape != inputs.shape:
                print('Targets shape is:', self.targets.shape)
                print('Inputs shape is:', inputs.shape)
                raise ValueError('Input and target shape do not match')
                
            # split off validation data
            self.message_sig.emit('Splitting data...')
            numIm = inputs.shape[0]
            val_inds = np.random.choice(np.arange(numIm),
                                        np.round(.2*numIm).astype(np.int),
                                        replace=False)
            valX = np.take(inputs,val_inds,axis=0)
            valY = np.take(self.targets,val_inds, axis=0)
            trainX = np.delete(inputs, val_inds, axis=0)
            trainY = np.delete(self.targets, val_inds, axis=0)
            
            # calculate number of epochs and batches
            numEp = np.minimum(np.int(1*(self.FNind+1)),70)
            batchSize = 8
            numBatches = trainX.shape[0]*numEp/batchSize
            
            # Make progress callback
            progCB = ProgressCallback()
            progCB.thread = self
            progCB.progress = 0
            progCB.batchpercent = 100/numBatches
            self.CBs = self.CBs + [progCB]
            
            self.message_sig.emit('Training model...')
            with self.graph.as_default():
                self.model.fit(x=trainX, y=trainY, batch_size=batchSize,
                       epochs=numEp, shuffle=True,
                       validation_data=(valX,valY),
                       verbose = 0,
                       callbacks=self.CBs)
            
            self.batch_sig.emit(100)    
            
            keras.backend.clear_session()
            keras.backend.tf.reset_default_graph()
                
            self.message_sig.emit('Training Complete.')
            self.message_sig.emit('Loading best model...')
            graph = keras.backend.tf.get_default_graph()
            with graph.as_default():
                self.model = load_model(self.model_path,custom_objects={'dice_loss':dice_loss})
            
            
            self.message_sig.emit('Evaluating on validation data...')
            score = self.model.evaluate(valX,valY,verbose=0)
            self.message_sig.emit('Dice Score: {:.2f}'.format(1-score))
            
            self.model_sig.emit(self.model, graph)
            
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
    
    def __init__(self,graph,ims,model):
        QThread.__init__(self)
        self.graph = graph
        self.ims = ims
        self.model = model
        
    def __del__(self):
        self.wait()

    def run(self):
        try:
            inputs = self.ims[...,np.newaxis]
            
            with self.graph.as_default():
                output = self.model.predict(inputs,batch_size=16,verbose=0)
                
            mask = (output[...,0]>.5).astype(np.float)
                    
            self.segmask_sig.emit(mask)
        except Exception as e:
            self.error_sig.emit(str(e))
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
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