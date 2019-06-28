from keras.layers import Input,Conv3D,BatchNormalization,concatenate
from keras.layers import ELU, GlobalAveragePooling3D, ZeroPadding3D, Dense
from keras.models import Model

def ClassModel3D(input_shape=((256,256,64,1)),numBlocks=3,filt_mult=8,num_classes=1):
    inp = Input(shape=(input_shape),name='input_layer')
    x = inp

    for rr in range(numBlocks):
        x1 = Conv3D(filt_mult*(2**(rr)), (1, 1, 1),padding='same',name='Conv1_{}'.format(rr))(x)
        x1 = ELU(name='elu_{}_1'.format(rr))(x1)
        x3 = Conv3D(filt_mult*(2**(rr)), (3, 3, 3),padding='same',name='Conv3_{}'.format(rr))(x)
        x3 = ELU(name='elu_{}_3'.format(rr))(x3)
        x51 = Conv3D(filt_mult*(2**(rr)), (3, 3, 3),padding='same',name='Conv51_{}'.format(rr))(x)
        x51 = ELU(name='elu_{}_51'.format(rr))(x51)
        x52 = Conv3D(filt_mult*(2**(rr)), (3, 3, 3),padding='same',name='Conv52_{}'.format(rr))(x51)
        x52 = ELU(name='elu_{}_52'.format(rr))(x52)
        x = concatenate([x1,x3,x52],name='merge_{}'.format(rr))
        x = Conv3D(filt_mult*(2**(rr)),(1,1,1),padding='valid',name='ConvAll_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_{}'.format(rr))(x)
        x = ZeroPadding3D(padding=(1,1,1),name='PrePad_{}'.format(rr))(x)
        if x.shape[3] >= 8:
            x = Conv3D(filt_mult*(2**(rr)),(4,4,4),padding='valid',strides=(2,2,2),name='ConvStride_{}'.format(rr))(x)
        else:
            x = Conv3D(filt_mult*(2**(rr)),(4,4,3),padding='valid',strides=(2,2,1),name='ConvStride_{}'.format(rr))(x)
        x = ELU(name='elu{}_2'.format(rr))(x)

    x = GlobalAveragePooling3D()(x)
    if num_classes > 1:
        x = Dense(num_classes,activation='softmax')(x)
    elif num_classes == 1:
        x = Dense(num_classes,activation='sigmoid')(x)
    return Model(inputs=inp,outputs=x)

if __name__ == '__main__':
    input_shape = (256,256,44,1)
    numBlocks = 4
    filt_mult = 8
    num_classes = 1
    model = ClassModel3D(input_shape,numBlocks,filt_mult,num_classes)
    model.summary()