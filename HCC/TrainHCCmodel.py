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
from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from HCC_Models import BlockModel_Classifier, Inception_model, ResNet50
from HelperFunctions import CyclicLR, GetDatagens, LoadValData

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
    im_dims = (256, 256)
    incl_channels = ['T1p', 'T1a', 'T1v']
    # incl_channels = ['Inp','Out','T2f','T1p','T1a','T1v','T1d','Dw1','Dw2']
    n_channels = len(incl_channels)
    batch_size = 8
    epochs = 20
    multi_process = False
    best_weights_file = 'HCC_best_model_weights_blockmodel.h5'
    # model_weight_path = 'HCC_classification_model_weights_{epoch:02d}-{val_acc:.4f}.h5'
    model_weight_path = 'HCC_best_validation_weights.h5'
    val_split = .2

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
        pos_dir, neg_dir, batch_size, im_dims, incl_channels)

    # Load validation data
    x_val, y_val = LoadValData(pos_dir, neg_dir, im_dims, incl_channels)

    # Test datagens
    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(testX[2,...,i],cmap='gray')
    # plt.show()

    # Setup model
    # HCCmodel = ResNet50(input_shape=im_dims+(n_channels,), classes=1)
    HCCmodel = Inception_model(input_shape=im_dims+(n_channels,))
    # HCCmodel = BlockModel_Classifier(
    #     im_dims+(n_channels,), filt_num=8, numBlocks=6)
    HCCmodel.summary()

    # compile
    HCCmodel.compile(Adam(lr=1e-5), loss=binary_crossentropy,
                     metrics=['accuracy'])
    # HCCmodel.compile(SGD(lr=1e-5,momentum=.8),loss=binary_crossentropy,metrics=['accuracy'])

    if resume:
        HCCmodel.load_weights(best_weights_file)

    # Setup callbacks
    cb_check = ModelCheckpoint(model_weight_path, monitor='val_loss', verbose=1,
                               save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # cb_plateau = ReduceLROnPlateau(
    #     monitor='val_loss', factor=.5, patience=5, verbose=1)
    # cb_tb = TensorBoard(log_dir='logs/{}'.format(time()),
    #                     histogram_freq=0, batch_size=8,
    #                     write_grads=False,
    #                     write_graph=False)
    clr_step_size = len(train_gen.list_IDs)/batch_size*4
    cb_clr = CyclicLR(base_lr=1e-5, max_lr=2e-4,
                      step_size=clr_step_size, mode='triangular')

    # Train model
    print('Starting training...')
    history = HCCmodel.fit_generator(generator=train_gen,
                                     epochs=epochs, class_weight=class_weight_dict,
                                     use_multiprocessing=multi_process,
                                     workers=8, verbose=1, callbacks=[cb_check, cb_clr],
                                     validation_data=(x_val, y_val))
    # Rename best weights
    h5files = glob('*.h5')
    load_file = max(h5files, key=os.path.getctime)
    if os.path.exists(best_weights_file):
        os.remove(best_weights_file)
    os.rename(load_file, best_weights_file)
    print('Renamed {} to {}'.format(load_file, best_weights_file))

    # Load best weights
    HCCmodel.load_weights(best_weights_file)

    print('Calculating classification confusion matrix...')
    preds = HCCmodel.predict(x_val, batch_size=batch_size, verbose=1)
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
    print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
    print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
    print('% Sensitivity: {:.02f}'.format(100*(tp)/(tp+fn)))
    print('% Specificity: {:.02f}'.format(100*(tn)/(tn+fn)))
    print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
    print('-----------------------')

    # with open('val_results.txt', 'w') as f:
    #     for i, j in zip(val_gen.list_IDs[:totalNum], preds[:, 0]):
    #         f.write("{}-{:.04f}\n".format(i, j))
    # print('Wrote validation results to file')
