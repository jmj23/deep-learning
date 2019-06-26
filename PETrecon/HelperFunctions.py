#%% Model
from keras.layers import Input, Conv2D, concatenate, Conv3D#, Cropping2D
from keras.layers import BatchNormalization, Conv2DTranspose, ZeroPadding2D
from keras.layers import UpSampling2D, Reshape
from keras.layers.advanced_activations import ELU
from keras.models import Model
def BlockModel_reg(samp_input, dual_output, fnum=8):
    lay_input = Input(shape=(samp_input.shape[1:]), name='input_layer')
    padamt = 1
    MSconv = Conv3D(16, (3, 3, 3), padding='valid', name='MSconv')(lay_input)
    bn = BatchNormalization()(MSconv)
    MSact = ELU(name='MSelu')(bn)
    MSconvRS = Reshape((254, 254, 16))(MSact)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(fnum*(2**(rr-1)), (1, 1), padding='same',
                       name='Conv1_{}'.format(rr))(MSconvRS)
    lay_conv3 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                       name='Conv3_{}'.format(rr))(MSconvRS)
    lay_conv51 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                        name='Conv51_{}'.format(rr))(MSconvRS)
    lay_conv52 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                        name='Conv52_{}'.format(rr))(lay_conv51)
    lay_merge = concatenate([lay_conv1, lay_conv3, lay_conv52],
                            name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(fnum*(2**(rr-1)), (1, 1), padding='valid',
                          name='ConvAll_{}'.format(rr))(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(fnum*(2**(rr-1)), (4, 4), padding='valid', strides=(2, 2),
                        name='ConvStride_{}'.format(rr))(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    # contracting blocks 2-3
    for rr in range(2, 4):
        lay_conv1 = Conv2D(fnum*(2**(rr-1)), (1, 1), padding='same',
                           name='Conv1_{}'.format(rr))(lay_act)
        lay_conv3 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                           name='Conv3_{}'.format(rr))(lay_act)
        lay_conv51 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                            name='Conv51_{}'.format(rr))(lay_act)
        lay_conv52 = Conv2D(fnum*(2**(rr-1)), (3, 3), padding='same',
                            name='Conv52_{}'.format(rr))(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],
                                name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(fnum*(2**(rr-1)),(1, 1), padding='valid',
                              name='ConvAll_{}'.format(rr))(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(fnum*(2**(rr-1)), (4, 4), padding='valid', strides=(2, 2),
                            name='ConvStride_{}'.format(rr))(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    # expanding block 3
    dd = 3
    lay_deconv1 = Conv2D(fnum*(2**(dd-1)),(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_act)
    lay_deconv3 = Conv2D(fnum*(2**(dd-1)),(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_act)
    lay_deconv51 = Conv2D(fnum*(2**(dd-1)), (3,3),padding='same',name='DeConv51_{}'.format(dd))(lay_act)
    lay_deconv52 = Conv2D(fnum*(2**(dd-1)), (3,3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(fnum*(2**(dd-1)),(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)
    
    lay_cleanup = Conv2DTranspose(fnum*(2**(dd-1)), (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(fnum*(2**(dd-1)), (3,3), padding='same', name='cleanup{}_2'.format(dd))(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # expanding blocks 2-1
    expnums = list(range(1,3))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(fnum*(2**(dd-1)),(1,1),padding='same',name='DeConv1_{}'.format(dd))(lay_skip)
        lay_deconv3 = Conv2D(fnum*(2**(dd-1)),(3,3),padding='same',name='DeConv3_{}'.format(dd))(lay_skip)
        lay_deconv51 = Conv2D(fnum*(2**(dd-1)), (3, 3),padding='same',name='DeConv51_{}'.format(dd))(lay_skip)
        lay_deconv52 = Conv2D(fnum*(2**(dd-1)), (3, 3),padding='same',name='DeConv52_{}'.format(dd))(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(fnum*(2**(dd-1)),(1,1),padding='valid',name='DeConvAll_{}'.format(dd))(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(fnum*(2**(dd-1)), (3, 3),name='cleanup{}_1'.format(dd))(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(fnum*(2**(dd-1)), (3,3), padding='same',name='cleanup{}_2'.format(dd))(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
        
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
        
    # regressor
    lay_reg = Conv2D(1,(1,1), activation='linear',name='reg_output')(lay_pad)
    if dual_output:
        # classifier
        lay_class = Conv2D(4,(1,1), activation='softmax',name='class_output')(lay_pad)
        returnModel = Model(inputs=[lay_input],outputs=[lay_reg,lay_class])
    else:
        returnModel = Model(lay_input,lay_reg)
        
    return returnModel