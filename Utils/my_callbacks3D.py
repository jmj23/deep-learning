import keras
from matplotlib import pyplot as plt

class Plots(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_epoch_end(self, epoch, logs={}):
        
        return
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        prd = self.model.predict(self.test,batch_size=1)
    
        self.imobj.set_data(prd[0,15,:,:,0])
        plt.pause(.005)
        plt.draw()
        return
    

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        return

    def on_epoch_end(self, epoch, logs={}):
        
        return
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return