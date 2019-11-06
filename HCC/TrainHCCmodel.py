# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-member
import os
import sys
from glob import glob
from os.path import join
from time import time

import GPUtil
import keras.backend as K
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam
from keras_radam import RAdam
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image

from HCC_Models import BlockModel_Classifier, Inception_model, ResNet50, SmallModel
from HelperFunctions import CyclicLR, GetDatagens, LoadValData, focal_loss

K.set_image_data_format('channels_last')


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['HDF5_USE_FILE_LOCKING'] = 'false'

    try:
        if not 'DEVICE_ID' in locals():
            DEVICE_ID = GPUtil.getFirstAvailable()[0]
            print('Using GPU', DEVICE_ID)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    except Exception as e:
        print('No GPU available')
        print('Using CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Datapaths
    # datapath = os.path.expanduser(join('~', 'deep-learning', 'HCC', 'Data'))
    datapath = join('D:\\', 'jmj136', 'HCCdata')

    # parameters
    randseed = 2
    im_dims = (384, 384)
    # incl_channels = ['T1p', 'T1a', 'T1v']
    incl_channels = ['Inp','Out','T2f','T1p','T1a','T1v','T1d']
    n_channels = len(incl_channels)
    batch_size = 16
    multi_process = False
    # model_weight_path = 'HCC_classification_model_weights_{epoch:02d}-{val_acc:.4f}.h5'
    model_weight_path = 'HCC_weights_{}_channels_v1.h5'.format(n_channels)
    val_split = .2

    # [warm up epochs, normal epochs]
    epochs = [10,500]

    # RESUME if needed
    # resume = True
    resume = False

    # allow for command-line epochs argument
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except Exception as e:
            pass

    # Get datagen
    pos_dir = join(datapath, 'Positive')
    neg_dir = join(datapath, 'Negative')
    train_gen, _, class_weight_dict = GetDatagens(
        pos_dir, neg_dir, batch_size, im_dims, incl_channels, randseed)


    # print some images to file for verification
    # tempX,tempY = train_gen.__getitem__(0)
    # for ind in np.arange(0,tempX.shape[0]):
    #     im = np.copy(tempX[ind,...,1])
    #     img = Image.fromarray((im*255).astype(np.uint8))
    #     label = tempY[ind]
    #     img.save('Verify_imgs/Train_img_{}_{}.png'.format(ind,label))

    # Load validation data
    x_val, y_val = LoadValData(pos_dir, neg_dir, im_dims, incl_channels, randseed)


    # Setup model
    # HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
    # HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))
    HCCmodel = BlockModel_Classifier(im_dims+(n_channels,), filt_num=8, numBlocks=6)
    # HCCmodel = SmallModel(im_dims + (n_channels,))
    # HCCmodel.summary()

    # compile
    opt = SGD(lr=1e-4,momentum=.8)
    # opt = Adam(lr=1e-4)
    # opt = RAdam(learning_rate=1e-6)
    loss = binary_crossentropy
    # loss = focal_loss(alpha=.25, gamma=2)

    HCCmodel.compile(opt,loss=loss,metrics=['accuracy'])

    if resume:
        HCCmodel.load_weights(model_weight_path)

    # Setup callbacks
    cb_check = ModelCheckpoint(model_weight_path, monitor='val_loss', verbose=1,
                               save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # cb_plateau = ReduceLROnPlateau(
    #     monitor='val_loss', factor=.5, patience=5, verbose=1)
    cb_tb = TensorBoard(log_dir='logs/{}'.format(time()),
                        histogram_freq=0, batch_size=8,
                        write_grads=False,
                        write_graph=False)
    # clr_step_size = len(train_gen.list_IDs)/batch_size*8
    # cb_clr = CyclicLR(base_lr=1e-7, max_lr=1e-5,
    #                   step_size=clr_step_size, mode='triangular2')

    # Train model
    print('Starting warmup training...')
    history = HCCmodel.fit_generator(generator=train_gen,
                                     epochs=epochs[0], class_weight=class_weight_dict,
                                     use_multiprocessing=multi_process,
                                     workers=8, verbose=1, callbacks=[cb_check, cb_tb],
                                     validation_data=(x_val, y_val))
    
    print('Learning rate was:')
    print(K.get_value(HCCmodel.optimizer.lr))
    # set learning rate
    K.set_value(HCCmodel.optimizer.lr, 5e-3)
    print('Learning rate is now:')
    print(K.get_value(HCCmodel.optimizer.lr))

    print('Resuming training...')
    history = HCCmodel.fit_generator(generator=train_gen,
                                     epochs=epochs[1], #class_weight=class_weight_dict,
                                     use_multiprocessing=multi_process,
                                     workers=8, verbose=1, callbacks=[cb_check, cb_tb],
                                     validation_data=(x_val, y_val))

    # Load best weights
    HCCmodel.load_weights(model_weight_path)

    print('Getting validation predictions and results...')
    preds = HCCmodel.predict(x_val, batch_size=batch_size, verbose=1)
    print('----------------------')
    print('Some predictions:')
    print(np.stack((preds[:5,0],y_val[:5]),axis=1))
    print(np.stack((preds[-5:,0],y_val[-5:]),axis=1))
    print('----------------------')
    print('Min: {:.02f}'.format(preds.min()))
    print('Max: {:.02f}'.format(preds.max()))
    print('Mean: {:.02f}'.format(preds.mean()))
    print('Std: {:.02f}'.format(preds.std()))
    y_pred = np.rint(preds)
    totalNum = len(y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    print('----------------------')
    print('Classification Results')
    print('----------------------')
    print('True positives: {}'.format(tp))
    print('True negatives: {}'.format(tn))
    print('False positives: {}'.format(fp))
    print('False negatives: {}'.format(fn))
    print('% Positive predicted: {:.02f}'.format(100*(tp+fp)/totalNum))
    print('% Negative predicted: {:.02f}'.format(100*(tn+fn)/totalNum))
    print('% Positive actual: {:.02f}'.format(100*(tp+fn)/totalNum))
    print('% Negative actual: {:.02f}'.format(100*(tn+fp)/totalNum))
    print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
    print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fn)))
    print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
    print('-----------------------')

    tfile = "HCC_model_results.txt"
    with open(tfile, "w") as text_file:
        text_file.write('Predictions:\n')
        text_file.write('Min: {:.02f}\n'.format(preds.min()))
        text_file.write('Max: {:.02f}\n'.format(preds.max()))
        text_file.write('Mean: {:.02f}\n'.format(preds.mean()))
        text_file.write('Std: {:.02f}\n'.format(preds.std()))
        text_file.write('----------------------\n')
        text_file.write('Classification Results\n')
        text_file.write('----------------------\n')
        text_file.write('True positives: {}\n'.format(tp))
        text_file.write('True negatives: {}\n'.format(tn))
        text_file.write('False positives: {}\n'.format(fp))
        text_file.write('False negatives: {}\n'.format(fn))
        text_file.write('% Positive predicted: {:.02f}\n'.format(100*(tp+fp)/totalNum))
        text_file.write('% Negative predicted: {:.02f}\n'.format(100*(tn+fn)/totalNum))
        text_file.write('% Positive actual: {:.02f}\n'.format(100*(tp+fn)/totalNum))
        text_file.write('% Negative actual: {:.02f}\n'.format(100*(tn+fp)/totalNum))
        text_file.write('% Sensitivity: {:.02f}\n'.format(100*(tp)/(tp+fn)))
        text_file.write('% Specificity: {:.02f}\n'.format(100*(tn)/(tn+fn)))
        text_file.write('% Accuracy: {:.02f}\n'.format(100*(tp+tn)/totalNum))
        text_file.write('-----------------------\n')

    print('Results written to {}'.format(tfile))