from __future__ import print_function

import glob
import nibabel
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import Unet

#K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# spatial resolution of the input images
img_rows = 128
img_cols = 128
# number of adjacent slices to include for improved 3d sensitivity (1, 3, or 5)
img_channels = 3
data_folder = '/home/axm3/data/deepTx/inputs/train'
#data_folder = '/Users/axm3/Documents/data/deepLearning/deepTx/CoregNOBONE_pad'

def resliceToAxial( data ):
    return np.transpose( data, (2,0,1) )
def resliceToCoronal( data ):
    return np.transpose( data, (1,0,2) )
def resliceToSagittal( data ):
    return np.transpose( data, (0,1,2) )

def get_myUnet():
    model = Unet.UNetContinuous([img_rows,img_cols,img_channels],out_ch=1,start_ch=16,depth=7,inc_rate=2.,activation='relu',dropout=0.5,batchnorm=True,maxpool=True,upconv=True,residual=False)
    model.compile(optimizer=Adam(lr=1e-5), loss=mean_squared_error, metrics=[mean_squared_error,mean_absolute_error])
    return model


def train():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    for network_type in range(0, 3):
        for slice_orient in range(0, 1):
            
            np.random.seed(813)
                        
            if slice_orient == 0:
                print('-'*50)
                print('AXIAL NETWORK...')
                print('-'*50)
                weightfile_slicename = 'axi'
            elif slice_orient == 1:
                print('-'*50)
                print('CORONAL NETWORK...')
                print('-'*50)
                weightfile_slicename = 'cor'
            else:
                print('-'*50)
                print('SAGITTAL NETWORK...')
                print('-'*50)
                weightfile_slicename = 'sag'

            print('-'*50)
            print('Creating and compiling network...')
            print('-'*50)

            weightfile_networkname = 'unetLrg' + repr(img_channels) + 'ch'
            model = get_myUnet()

            weightfile_name = 'weights_' + weightfile_networkname + '_' + weightfile_slicename + '.h5'
            model_checkpoint = ModelCheckpoint(weightfile_name, monitor='loss', save_best_only=True)

            print('-'*50)
            print('Loading images from ' + data_folder + '...')
            print('-'*50)
            img1_files = glob.glob(os.path.join(data_folder,'T1_2mm_MNI_pdt*.gz'))
            img_count = 0
            for curr_img1_file in img1_files:
                img_count += 1
        
                # read in T1 image first
                curr_img1_nii = nibabel.load(curr_img1_file)
        
                print( curr_img1_file )
        
                # get filename of matching Tx image
                subj_id_suffix = curr_img1_file.split('pdt')
                curr_img2_file = os.path.join(data_folder,'Tx_MNI_pdt' + subj_id_suffix[1])
                curr_img2_nii = nibabel.load(curr_img2_file)
                
                # data reslicing
                if slice_orient == 0: # axial
                    curr_img1 = resliceToAxial(curr_img1_nii.get_data())
                    curr_img2 = resliceToAxial(curr_img2_nii.get_data())
                elif slice_orient == 1:
                    curr_img1 = resliceToCoronal(curr_img1_nii.get_data())
                    curr_img2 = resliceToCoronal(curr_img2_nii.get_data())
                else:
                    curr_img1 = resliceToSagittal(curr_img1_nii.get_data())
                    curr_img2 = resliceToSagittal(curr_img2_nii.get_data())

                # data normalization for T1 input (Tx images are normalized later, below)
                curr_img1 = curr_img1.astype('float32')
                mean_img1 = np.mean(curr_img1)
                std_img1 = np.std(curr_img1)
                curr_img1 -= mean_img1
                curr_img1 /= std_img1

                # if 1 image channel, just take the slice as is
                if img_channels == 1:
                    curr_img1_channels = curr_img1
                # if 3 image channels, set channels as three adjacent slices
                elif img_channels == 3:
                    curr_img1_channels = np.zeros( (curr_img1.shape[0], curr_img1.shape[1],curr_img1.shape[2], 3) )
                    for curr_slice in range(0, curr_img1.shape[0]-1):
                        curr_img1_channel1 = curr_img1[curr_slice,:,:]
                        if ( curr_slice - 1 < 0 ):
                            curr_img1_channel0 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                            curr_img1_channel2 = curr_img1[curr_slice+1,:,:]
                        elif ( (curr_slice + 1) > (curr_img1.shape[0]-1) ):
                            curr_img1_channel0 = curr_img1[curr_slice-1,:,:]
                            curr_img1_channel2 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                        else:
                            curr_img1_channel0 = curr_img1[curr_slice-1,:,:]
                            curr_img1_channel2 = curr_img1[curr_slice+1,:,:]
                            
                        curr_img1_channels[curr_slice,:,:,0] = curr_img1_channel0
                        curr_img1_channels[curr_slice,:,:,1] = curr_img1_channel1
                        curr_img1_channels[curr_slice,:,:,2] = curr_img1_channel2
                # if 5 image channels, set channels as three adjacent slices
                elif img_channels == 5:
                    curr_img1_channels = np.zeros( (curr_img1.shape[0], curr_img1.shape[1],curr_img1.shape[2], 5) )
                    for curr_slice in range(0, curr_img1.shape[0]-1):
                        curr_img1_channel2 = curr_img1[curr_slice,:,:]
                        if ( curr_slice - 2 < 0 ):
                            curr_img1_channel0 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                            curr_img1_channel1 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                            curr_img1_channel3 = curr_img1[curr_slice+1,:,:]
                            curr_img1_channel4 = curr_img1[curr_slice+2,:,:]
                        elif ( curr_slice - 1 < 0 ):
                            curr_img1_channel0 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                            curr_img1_channel1 = curr_img1[curr_slice-1,:,:]
                            curr_img1_channel3 = curr_img1[curr_slice+1,:,:]
                            curr_img1_channel4 = curr_img1[curr_slice+2,:,:]
                        elif ( (curr_slice + 1) > (curr_img1.shape[0]-1) ):
                            curr_img1_channel0 = curr_img1[curr_slice-2,:,:]
                            curr_img1_channel1 = curr_img1[curr_slice-1,:,:]
                            curr_img1_channel3 = curr_img1[curr_slice+1,:,:]
                            curr_img1_channel4 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                        elif ( (curr_slice + 2) > (curr_img1.shape[0]-1) ):
                            curr_img1_channel0 = curr_img1[curr_slice-2,:,:]
                            curr_img1_channel1 = curr_img1[curr_slice-1,:,:]
                            curr_img1_channel3 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                            curr_img1_channel4 = np.zeros( (curr_img1.shape[1], curr_img1.shape[2]) )
                        else:
                            curr_img1_channel0 = curr_img1[curr_slice-2,:,:]
                            curr_img1_channel1 = curr_img1[curr_slice-1,:,:]                            
                            curr_img1_channel3 = curr_img1[curr_slice+1,:,:]
                            curr_img1_channel4 = curr_img1[curr_slice+2,:,:]      
                            
                        curr_img1_channels[curr_slice,:,:,0] = curr_img1_channel0
                        curr_img1_channels[curr_slice,:,:,1] = curr_img1_channel1
                        curr_img1_channels[curr_slice,:,:,2] = curr_img1_channel2
                        curr_img1_channels[curr_slice,:,:,3] = curr_img1_channel3
                        curr_img1_channels[curr_slice,:,:,4] = curr_img1_channel4
            
                if img_count == 1:
                    img1 = curr_img1_channels
                    img2 = curr_img2
                else:
                    # stack in 3rd dimension
                    img1 = np.concatenate( ( img1, curr_img1_channels ), axis=0 )
                    img2 = np.concatenate( ( img2, curr_img2 ), axis=0 )
    
            print(repr(img_count) + ' files loaded')
    
            # make img2 '4D' with one channel
            img2=np.expand_dims(img2,3)
        
            print('-'*50)
            print('Data augmentation...')
            print('-'*50)
        
            # data centering for Tx
            img2 = img2.astype('float32')
            mean_img2 = np.mean(img2)
            std_img2 = np.std(img2)
            img2 -= mean_img2
            img2 /= std_img2
            np.save('rescalers.npy',[mean_img2,std_img2])
            print('img2 mean: ' + repr(mean_img2) + ' std: ' + repr(std_img2))
        
            datagen = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest')
            # note that datagen.fit is not required here for these options
        
            print('-'*50)
            print('Fitting network...')
            print('-'*50)
        
            bs = 32
            ep = 30
            model.fit_generator( datagen.flow( img1, img2, batch_size=bs ), steps_per_epoch=1000, epochs=ep, verbose=1, callbacks=[model_checkpoint] )
            #model.fit(img1, img2, batch_size=32, epochs=30, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])


if __name__ == '__main__':
    train()
