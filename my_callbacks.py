import keras
from matplotlib import pyplot as plt
import numpy as np

class Plots(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        test = np.reshape(self.test,np.r_[1,self.test.shape])
        prd = self.model.predict(test,batch_size=1)
    
        self.imobj.set_data(prd[0,:,:,0])
        plt.pause(.005)
        plt.draw()
        return
    
    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        prd = self.model.predict(self.test,batch_size=1)
#    
#        self.imobj.set_data(prd[0,:,:,0])
#        plt.pause(.005)
#        plt.draw()
        return
    

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.dice = []
        self.jac = []
        self.val_loss = []
        self.val_dice = []
        self.val_jac = []
        return
    
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef'))
        self.jac.append(logs.get('jac_met'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_dice.append(logs.get('val_dice_coef'))
        self.val_jac.append(logs.get('val_jac_met'))
        return
    
class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        prd = self.model.predict(self.test,batch_size=1)
        prd_r = np.round(prd)
        targ = self.target
        intersection = prd_r*targ
        union = np.maximum(prd_r, targ)
        intsum = np.sum(intersection)
        unsum = np.sum(union)
        print("Jac=",intsum/unsum)
        