'''Code demonstration for introductory deep learning programming
in Keras. This tutorial walks through an example in segmentation
'''

#%% Part 1: Classification
#%% Initial Preparation
# First, import necessary modules
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt

# Set some parameters
batch_size = 128
num_classes = 10
epochs = 2

#%% Data preparation
# As with all deep learning applications, we need to do some data
# preparation first
# For our segmentation application, we will be segmenting
# images that have 4 digits in them. The segmentation will be
# semantic, in that we want to say if each pixel belongs to 
# a 0, 1, 2, 3,...
# Indeed, semantic segmentation is simply classification at the pixel level

# We first need to rearrange our data into these 4-digit images
# We will make them square- that is, 2 by 2 digits
# There's many ways to do this, but the most transparent way is a for loop

# Since we will need to apply this to both the training and testing data,
# let's make a function that can be used on both
def ConvertXdata(x_data):
    # pre allocate
    newSampNum = np.round(x_data.shape[0]/4).astype(np.int)
    x_out = np.zeros((newSampNum,2*28,2*28))
    for ii in range(0,x_data.shape[0],4):
        samp = np.round(ii/4).astype(np.int)
        x_out[samp,:28,:28] = x_data[ii,...,0]
        x_out[samp,28:,:28] = x_data[ii+1,...,0]
        x_out[samp,28:,28:] = x_data[ii+2,...,0]
        x_out[samp,:28,28:] = x_data[ii+3,...,0]
    return x_out[...,np.newaxis]

# Now we can apply this function
x_train2 = ConvertXdata(x_train)    
x_test2 = ConvertXdata(x_test)

# And while we're at it, let's get the input shape for setting
# up our model later
input_shape2 = x_train2.shape[1:4]

# Let's display one of each to make sure we did it correctly
testnum = 20
fig1 = plt.figure(1)
plt.imshow(x_train2[testnum,...,0],cmap='gray')
fig2 = plt.figure(2)
plt.imshow(x_test2[testnum,...,0],cmap='gray')

# Looks great! But deep learning needs both inputs and targets...
# so we have to make targets for our segmentation

# Again, there are a variety of ways to go about this
# A straightforward way is to first create the labeled
# images, then reshape them into 2x2 digits.

# Again, we'll create a function for this since we
# have both training and testing targets
def ConvertYdata(x_data,y_data):
    seg = np.zeros(x_data.shape[:-1])
    for ii in range(0,seg.shape[0]):
        curim = x_data[ii,...,0]
        curim[curim>0.1] = np.argmax(y_data[ii])+1
        seg[ii,...] = curim
        
    newSampNum = np.round(x_data.shape[0]/4).astype(np.int)
    y_out = np.zeros((newSampNum,2*28,2*28))
    for ii in range(0,x_data.shape[0],4):
        samp = np.round(ii/4).astype(np.int)
        y_out[samp,:28,:28] = seg[ii]
        y_out[samp,28:,:28] = seg[ii+1]
        y_out[samp,28:,28:] = seg[ii+2]
        y_out[samp,:28,28:] = seg[ii+3]
        
    return keras.utils.to_categorical(y_out, num_classes+1)
    
# Apply the function to train and test...
y_train2 = ConvertYdata(x_train,y_train)
y_test2 = ConvertYdata(x_test,y_test)

# It's import to make sure our inputs match up with our targets
# Let's display them side by side to make sure it's right
testnum = 8
fig3 = plt.figure(3)
plt.imshow(np.c_[x_train2[testnum,...,0],np.argmax(y_train2[testnum],axis=2)-1],cmap='gray')
fig4 = plt.figure(4)
plt.imshow(np.c_[x_test2[testnum,...,0],np.argmax(y_test2[testnum],axis=2)-1],cmap='gray')

# So far so good! Our data is now ready for training!

#%% Building a segmentation network
# When we did the classification scheme, we ended up with a tiny, subsampled
# image that was then flattened for the final neural network steps
# For segmenation, we need to have classification for each pixel, so we must
# do something to counteract the subsampling that occurs with convolutions
# Introducing: Transposed Convolutions
from keras.layers import Conv2DTranspose
# Transpose convoutions do just the opposite of a regular convolution
# So we will use these layers to regain the spatial resolution we need 
# for pixel-wise classification

# With that in mind, let's build our segmentation model!
# We'll start the same way:
model2 = Sequential()
# Fewer kernels this time since our model needs to be bigger
model2.add(Conv2D(10, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape2))
model2.add(Conv2D(20, kernel_size=(3,3),
                  activation='relu'))
# Here's our strided downsampling layer
# We use a 4x4 kernel so that the image sizing
# works out in the end
model2.add(Conv2D(20, kernel_size=(4,4),
                 strides=(2,2),
                 activation='relu'))
model2.add(Conv2D(30, kernel_size=(3,3),
                  activation='relu'))
# This ends the "encoding" or "contracting" side of the network.
# Next, we will regain the lost spatial resolution with the matching
# "Decoding" or "expanding" side of the network

# Same layers in reverse order- just tranpose convolutions
model2.add(Conv2DTranspose(30,kernel_size=(3,3),
                           activation='relu'))
model2.add(Conv2DTranspose(30,kernel_size=(3,3),
                          strides=(2,2),
                          activation='relu'))
model2.add(Conv2DTranspose(20,kernel_size=(4,4),
                           activation='relu'))
# Final layer: use 11 filters to correspond to our 11 classes
# and use softmax activation
model2.add(Conv2DTranspose(11,kernel_size=(3,3),
                           activation='softmax'))
# Let's print out a summary of the model to make sure it's what we want.
model2.summary()
# The final output shape is the size of our image with the correct
# number of classes, 56x56x11. Perfect!
# Note: "none" means not fixed. The batch size hasn't been set yet
# so the first dimension doesn't have a fixed size

#%% Model compiling and training
# The rest of the steps are pretty much the same
# Compile the model, per usual
# This time we won't have the accuracy metric
# since that doesn't make as much sense here
model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam())

# And run the fitting
# Let's change our batch size
# since our inputs are 4 times larger
batch_size2 = 32
epochs2 = 2

model2.fit(x_train2, y_train2,
          batch_size=batch_size2,
          epochs=epochs2,
          verbose=1,
          validation_data=(x_test2, y_test2))
# The loss in this case is difficult to interpret.
# Let's see some results!

# We'll display the prediction and truth next to each other
# and see how it faired
predictions2 = model2.predict_classes(x_test2[:2])
fig5 = plt.figure(5)
plt.imshow(np.c_[predictions2[0]-1,np.argmax(y_test2[0],axis=2)-1],cmap='gray')
# Ooh... not great. Results may vary from
# simply mediocre to downright terrible
# It turn out that segmenting an image 4 times larger
# is much more difficult than simply classifying
# the small images.
# There are a variety of directions to go from here

# A deeper net gives more representational power
# to the model. If the problem is too complex for the current
# network, making it deeper should improve performance

# Some mathematical tricks, like batch normalization and
# ELU activations can help with the learning process
# and make the model learn quicker

# In segmentation, a particularly useful trick is the use of
# skip connetions, in which layers from the downsampling part
# of the network are concatenated with layers on the
# upsampling part. This both boosts the representational power
# of the model as well as improves the gradient flow, which
# also helps the model learn quicker.

# However, in this case, our segmentation is still pretty simple.
# I ran the training for 30 epochs on a GPU and the results were
# much, much better. 

# Happy (deep) learning!

#%% Part 3: Functional API
# So far, we've been what's called the 'Sequential' API
# (Application Programming Interface), which is OK for
# simple models. Basically, it means that our network
# has a single, straight path, ie 
# input -> convolution -> activation -> convolution -> output
# Each layer has a single input and output
#
# But what if we wanted something more complicated? What if
# we had two different inputs, for example? Then we would want
# something like
# input1 --> concatenate -> convolution -> ...
# input2 ----^
# Or maybe, what I mentioned as an alteration to the segmentation
# network:
# input -> conv1 -> conv2 -> deconv1 -> concatenate -> deconv2 -> output
#               `----------------------^
# The extra connection shown is called a skip connection.
# Skip connections allow the model to consider features that were calculated
# earlier in the network again, merged with further processed features
# in practice, this has shown to be hugely helpful in geting precise
# localization in segmentation outputs
# To make networks with this more complicated structure, we can't use
# the Sequential API. We need the Functional API.

# There are many, many reasons to want to use the functional API
# However, we will focus on the segmentation application as before,
# and show how a simple tweak in the functional API will give us
# significantly better results.

# We'll use the same segmentation data so no need to prepare anything new.
# Let's jump into model creation.


#%% Build a segmentation model with skip connections

# The functional model is just called "Model"
from keras.models import Model

# Some other layers we will need for this model
from keras.layers import Input, concatenate

# Creating a model in this way takes 2 arguments:
# Inputs
# Outputs
# So, we follow a process like this:
# -Define our input layer(s)
# -Create the rest of our network connected to those inputs
# -When we get to our final output layer(s), provide those
#   and the input(s) to "Model", and the result will be our
#   functional model!

# Our first layer will be an "Input layer", which is where
# we define the input shape we will be providing during training

inp = Input(shape=input_shape2)

# Right now, 'inp' defines our input layer. In the 
# next step, we will provide 'inp' as an argument to our
# next layer, which will then be called 'x1'. These variables
# don't matter too much. What matters is connecting the layers
# together in the proper order.
# Adding our first convolutional layer looks like this:
x1 = Conv2D(10,kernel_size=(3,3),activation='relu')(inp)
# Notice that we don't need to define the input shape for this
# layer, since we just did that in the Input layer.

# Let's build the rest of the encoding side of the network
x2 = Conv2D(20, kernel_size=(3,3),
                  activation='relu')(x1)
# Use kernel size of (2,2) to make matching up layers easier later
x3 = Conv2D(20, kernel_size=(2,2),
                 strides=(2,2),
                 activation='relu')(x2)
x4 = Conv2D(30, kernel_size=(3,3),
                  activation='relu')(x3)

# Now for the decoding side of the network, we will put the
# functional API to use by including skip connections
# The first layer is the same as usual. I'll add
# a 'd' for 'decoding'
x3d = Conv2DTranspose(30,kernel_size=(3,3),
                           activation='relu')(x4)
# Concatenate corresponding layers from the encoding and
# decoding sides of the network
# It can be tough to get layers to match up just right in size
# Playing around with kernel size and strides is usually needed
# so that concatenation can take place. The x,y spatial dimensions
# must be the same. Number of channels doesn't matter
cat = concatenate([x3d,x3])
# Now continue to add layers for the decoding side of the
# network, treating this merged layer like any other
x2d = Conv2DTranspose(30,kernel_size=(2,2),
                          strides=(2,2),
                          activation='relu')(cat)
cat = concatenate([x2d,x2])
# Notice that we can overwrite our previously set variable 'cat'
# without distrupting the network. The layers are still connected
# in the order we set them

x1d = Conv2DTranspose(20,kernel_size=(3,3),
                           activation='relu')(cat)
cat = concatenate([x1d,x1])
# Final output layer
out = Conv2DTranspose(11,kernel_size=(3,3),
                           activation='softmax')(cat)
# All of our layers and their connections are now defined. This is
# commonly referred to as a 'graph', where the layers are the nodes
# and the calculations from layers to layers are the edges.
# You can think of the information 'flowing' from the inputs to the outputs.
# Additionally, the information is stored as tensors, which are not defined
# arrays but rather placeholders for whichever data we feed in. Notice how
# we don't give our model any real data until the training begins?
# Now you know how "TensorFlow" got its name!

# To turn our graph into a real Keras model that we can use, simply call
# the 'Model' function we loaded
func_model = Model(inp,out)

# We can print out a summary of the model to make sure it's what we want.
# It's a little bit harder to keep track of layers in non-sequential
# format, but it's still a good way to make sure things look right.
func_model.summary()

# Now, everything else is just like the previous segmentation model
# Let's try it out and see how it works!
func_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam())
func_model.fit(x_train2, y_train2,
          batch_size=batch_size2,
          epochs=epochs2,
          verbose=1,
          validation_data=(x_test2, y_test2))

predictionsF = model2.predict_classes(x_test2[:2])
fig6= plt.figure(6)
plt.imshow(np.c_[predictionsF[0]-1,np.argmax(y_test2[0],axis=2)-1],cmap='gray')

# Well.... ok. It's about the same.
# However! In the long run (more than 2 epochs), having these skip connections
# will definitely make a difference. The difference becomes more pronounced
# for deeper networks (more layers) with more parameters and larger images.

# Now that you know the functional API, you can make any graph you like, train
# it, and use it! Once you've mastered the syntax and conceptual understanding
# of how to connect layers, you are only limited by your imagination as far
# as what kind of network you can build.

# Best of luck, and happy deep learning!