"""An implementation of the improved WGAN described in https://arxiv.org/abs/1704.00028

The improved WGAN has a term in the loss function which penalizes the network if its gradient
norm moves away from 1. This is included because the Earth Mover (EM) distance used in WGANs is only easy
to calculate for 1-Lipschitz functions (i.e. functions where the gradient norm has a constant upper bound of 1).

The original WGAN paper enforced this by clipping weights to very small values [-0.01, 0.01]. However, this
drastically reduced network capacity. Penalizing the gradient norm is more natural, but this requires
second-order gradients. These are not supported for some tensorflow ops (particularly MaxPool and AveragePool)
in the current release (1.0.x), but they are supported in the current nightly builds (1.1.0-rc1 and higher).

To avoid this, this model uses strided convolutions instead of Average/Maxpooling for downsampling. If you wish to use
pooling operations in your discriminator, please ensure you update Tensorflow to 1.1.0-rc1 or higher. I haven't
tested this with Theano at all.

The model saves images using pillow. If you don't have pillow, either install it or remove the calls to generate_images.
"""
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import h5py
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#%%

BATCH_SIZE = 8
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
#%%
# Model Save Path/name
model_savepath = 'LowDosePET_iWGAN_v1.hdf5'
# Data path/name
datapath = 'lowdosePETdata_v2.hdf5'

#%%
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.

    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.

    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)

#%%
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.

    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty

#%% Model
from keras.layers import concatenate, add, Lambda
#from keras.initializers import RandomNormal

def make_generator(input_shape):
    weight_init = 'glorot_uniform'#RandomNormal(mean=0., stddev=5*weight_std)
    
    lay_input = Input(shape=(input_shape),name='input_layer')
    
    padamt = 1
    crop = Cropping2D(cropping=((0, padamt), (0, padamt)), data_format=None)(lay_input)
    
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(16*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr),
                       kernel_initializer=weight_init)(crop)
    lay_conv3 = Conv2D(16*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr),
                       kernel_initializer=weight_init)(crop)
    lay_conv51 = Conv2D(16*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr),
                        kernel_initializer=weight_init)(crop)
    lay_conv52 = Conv2D(16*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr),
                        kernel_initializer=weight_init)(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(16*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr),
                          kernel_initializer=weight_init)(lay_merge)
    bn = BatchNormalization()(lay_conv_all)
    lay_act = ELU(name='elu{}_1'.format(rr))(bn)
    lay_stride = Conv2D(16*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr),
                        kernel_initializer=weight_init)(lay_act)
    lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    act_list = [lay_act]
    
    # contracting blocks 2-3
    for rr in range(2,3):
        lay_conv1 = Conv2D(16*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr),
                           kernel_initializer=weight_init)(lay_act)
        lay_conv3 = Conv2D(16*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr),
                           kernel_initializer=weight_init)(lay_act)
        lay_conv51 = Conv2D(16*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_act)
        lay_conv52 = Conv2D(16*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(16*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr),
                              kernel_initializer=weight_init)(lay_merge)
        bn = BatchNormalization()(lay_conv_all)
        lay_act = ELU(name='elu_{}'.format(rr))(bn)
        lay_stride = Conv2D(16*rr,(4,4),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
        act_list.append(lay_act)
    
    # expanding block 3
    dd=2
    lay_deconv1 = Conv2D(16*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd),
                         kernel_initializer=weight_init)(lay_act)
    lay_deconv3 = Conv2D(16*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd),
                         kernel_initializer=weight_init)(lay_act)
    lay_deconv51 = Conv2D(16*dd, (3,3),padding='same',name='DeConv51_{}'.format(dd),
                          kernel_initializer=weight_init)(lay_act)
    lay_deconv52 = Conv2D(16*dd, (3,3),padding='same',name='DeConv52_{}'.format(dd),
                          kernel_initializer=weight_init)(lay_deconv51)
    lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
    lay_deconv_all = Conv2D(16*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd),
                            kernel_initializer=weight_init)(lay_merge)
    bn = BatchNormalization()(lay_deconv_all)
    lay_act = ELU(name='elu_d{}'.format(dd))(bn)
    
    lay_up = UpSampling2D()(lay_act)
    
    lay_cleanup = Conv2DTranspose(16*dd, (3, 3),name='cleanup{}_1'.format(dd),
                                  kernel_initializer=weight_init)(lay_up)
    lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
    lay_cleanup = Conv2D(16*dd, (3,3), padding='same', name='cleanup{}_2'.format(dd),
                         kernel_initializer=weight_init)(lay_act)
    bn = BatchNormalization()(lay_cleanup)
    lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # expanding blocks 2-1
    expnums = list(range(1,2))
    expnums.reverse()
    for dd in expnums:
        lay_skip = concatenate([act_list[dd-1],lay_act],name='skip_connect_{}'.format(dd))
        lay_deconv1 = Conv2D(16*dd,(1,1),padding='same',name='DeConv1_{}'.format(dd),
                             kernel_initializer=weight_init)(lay_skip)
        lay_deconv3 = Conv2D(16*dd,(3,3),padding='same',name='DeConv3_{}'.format(dd),
                             kernel_initializer=weight_init)(lay_skip)
        lay_deconv51 = Conv2D(16*dd, (3, 3),padding='same',name='DeConv51_{}'.format(dd),
                              kernel_initializer=weight_init)(lay_skip)
        lay_deconv52 = Conv2D(16*dd, (3, 3),padding='same',name='DeConv52_{}'.format(dd),
                              kernel_initializer=weight_init)(lay_deconv51)
        lay_merge = concatenate([lay_deconv1,lay_deconv3,lay_deconv52],name='merge_d{}'.format(dd))
        lay_deconv_all = Conv2D(16*dd,(1,1),padding='valid',name='DeConvAll_{}'.format(dd),
                                kernel_initializer=weight_init)(lay_merge)
        bn = BatchNormalization()(lay_deconv_all)
        lay_act = ELU(name='elu_d{}'.format(dd))(bn)
        lay_up = UpSampling2D()(lay_act)        
        lay_cleanup = Conv2DTranspose(16*dd, (3, 3),name='cleanup{}_1'.format(dd),
                                      kernel_initializer=weight_init)(lay_up)
        lay_act = ELU(name='elu_cleanup{}_1'.format(dd))(lay_cleanup)
        lay_cleanup = Conv2D(16*dd, (3,3), padding='same',name='cleanup{}_2'.format(dd),
                             kernel_initializer=weight_init)(lay_act)
        bn = BatchNormalization()(lay_cleanup)
        lay_act = ELU(name='elu_cleanup{}_2'.format(dd))(bn)
    
    # regressor    
    lay_pad = ZeroPadding2D(padding=((0,2*padamt), (0,2*padamt)), data_format=None)(lay_act)
    lay_reg = Conv2D(1,(1,1), activation='linear',name='regression',
                     kernel_initializer=weight_init)(lay_pad)
    in0 = Lambda(lambda x : x[...,0],name='channel_split')(lay_input)
    in0 = Reshape([256,256,1])(in0)
    lay_res = add([in0,lay_reg],name='residual')
    
    return Model(lay_input,lay_res)

#%% Discriminator model
def make_discriminator(input_shape,test_shape,filtnum=16):
    
    weight_init = 'he_normal'#RandomNormal(mean=0., stddev=weight_std)
    
    # Conditional Inputs
    lay_input = Input(shape=input_shape,name='conditional_input')
    
    lay_step = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),name='StepdownLayer',
                      kernel_initializer=weight_init)(lay_input)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr),
                       kernel_initializer=weight_init)(lay_step)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr),
                       kernel_initializer=weight_init)(lay_step)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr),
                        kernel_initializer=weight_init)(lay_step)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr),
                        kernel_initializer=weight_init)(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr),
                          kernel_initializer=weight_init)(lay_merge)
    lay_act = ELU(name='elu{}_1'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr),
                        kernel_initializer=weight_init)(lay_act)
    lay_act1 = ELU(name='elu{}_2'.format(rr))(lay_stride)
    
    # Testing Input block
    lay_test = Input(shape=test_shape,name='test_input')
    lay_step2 = Conv2D(filtnum,(4,4),padding='valid',strides=(2,2),name='StepdownLayer2',
                       kernel_initializer=weight_init)(lay_test)
    # contracting block 1
    rr = 1
    lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}t'.format(rr),
                       kernel_initializer=weight_init)(lay_step2)
    lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}t'.format(rr),
                       kernel_initializer=weight_init)(lay_step2)
    lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}t'.format(rr),
                        kernel_initializer=weight_init)(lay_step2)
    lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}t'.format(rr),
                        kernel_initializer=weight_init)(lay_conv51)
    lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}t'.format(rr))
    lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}t'.format(rr),
                          kernel_initializer=weight_init)(lay_merge)
    lay_act = ELU(name='elu{}_1t'.format(rr))(lay_conv_all)
    lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}t'.format(rr),
                        kernel_initializer=weight_init)(lay_act)
    lay_act2 = ELU(name='elu{}_2t'.format(rr))(lay_stride)
    
    # Merge blocks
    lay_act = concatenate([lay_act1,lay_act2],name='InputMerge')
    # contracting blocks 2-5
    for rr in range(2,6):
        lay_conv1 = Conv2D(filtnum*rr, (1, 1),padding='same',name='Conv1_{}'.format(rr),
                           kernel_initializer=weight_init)(lay_act)
        lay_conv3 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv3_{}'.format(rr),
                           kernel_initializer=weight_init)(lay_act)
        lay_conv51 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv51_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_act)
        lay_conv52 = Conv2D(filtnum*rr, (3, 3),padding='same',name='Conv52_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_conv51)
        lay_merge = concatenate([lay_conv1,lay_conv3,lay_conv52],name='merge_{}'.format(rr))
        lay_conv_all = Conv2D(filtnum*rr,(1,1),padding='valid',name='ConvAll_{}'.format(rr),
                              kernel_initializer=weight_init)(lay_merge)
        lay_act = ELU(name='elu_{}'.format(rr))(lay_conv_all)
        lay_stride = Conv2D(filtnum*rr,(3,3),padding='valid',strides=(2,2),name='ConvStride_{}'.format(rr),
                            kernel_initializer=weight_init)(lay_act)
        lay_act = ELU(name='elu{}_2'.format(rr))(lay_stride)
    
    lay_flat = Flatten()(lay_act)
    lay_dense = Dense(1,activation='linear',name='realfake',
                      kernel_initializer=weight_init)(lay_flat)
    
    return Model(inputs=[lay_input,lay_test],outputs=[lay_dense])

#%%
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
#%%
if not 'x_train' in locals():
    print('Loading data...')
    with h5py.File(datapath,'r') as f:
        x_train = np.array(f.get('train_inputs'))[::4]
        y_train = np.array(f.get('train_targets'))[::4]
#        x_val = np.array(f.get('val_inputs'))
#        y_val = np.array(f.get('val_targets'))
        x_test = np.array(f.get('test_inputs'))
        y_test = np.array(f.get('test_targets')) 

# Now we initialize the generator and discriminator.
generator = make_generator(x_train.shape[1:])
discriminator = make_discriminator(x_train.shape[1:],y_train.shape[1:],16)

# The generator_model is used when we want to train the generator layers.
# As such, we ensure that the discriminator layers are not trainable.
# Note that once we compile this model, updating .trainable will have no effect within it. As such, it
# won't cause problems if we later set discriminator.trainable = True for the discriminator_model, as long
# as we compile the generator_model first.
for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_model = Model(inputs=generator.input, outputs=discriminator([generator.input,generator.output]))

# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=y_train.shape[1:])
cond_samples = Input(shape=x_train.shape[1:])
generated_samples_for_discriminator = generator.output
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator([cond_samples,real_samples])

# We also need to generate weighted-averages of real and generated samples, to use for the gradient norm penalty.
averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never really use the discriminator
# output for these samples - we're only running them to get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)

# The gradient penalty loss function requires the input averaged samples to get gradients. However,
# Keras loss functions can only have two arguments, y_true and y_pred. We get around this by making a partial()
# of the function with the averaged samples here.
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

# Keras requires that inputs and outputs have the same number of samples. This is why we didn't concatenate the
# real samples and generated samples before passing them to the discriminator: If we had, it would create an
# output with 2 * BATCH_SIZE samples, while the output of the "averaged" samples for gradient penalty
# would have only BATCH_SIZE samples.

# If we don't concatenate the real and generated samples, however, we get three outputs: One of the generated
# samples, one of the real samples, and one of the averaged samples, all of size BATCH_SIZE. This works neatly!
discriminator_model = Model(inputs=[real_samples, generator.output],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both the real and generated
# samples, and the gradient penalty loss for the averaged samples.
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

for epoch in range(100):
    np.random.shuffle(x_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(X_train.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                         [positive_y, negative_y, dummy_y]))
        generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE, 100), positive_y))
    # Still needs some code to display losses from the generator and discriminator, progress bars, etc.