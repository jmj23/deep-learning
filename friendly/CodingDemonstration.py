'''Code demonstration for introductory deep learning programming
in Keras. This demonstration has two parts:
1) Classification of MNIST dataset
2) Segmentation of digits
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

#%% Data loading and preparation

# Load the MNIST dataset
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# We'll just take half of the dataset to make our computations shorter
x_train = x_train[::2]
y_train = y_train[::2]
x_test = x_test[::2]
y_test = y_test[::2]

# Data is loaded as [samples,row,col]. Since it is grayscale, it has no
# channel data. Keras expects chanel data, so we add a singleton dimension
x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]
# When we make our Keras model, we'll need to give it input shape,
# so we'll define that here
# Keras just needs the row, column, and channel sizes
# With Python 0-indexing, that is axes 1, 2 and 3
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
# Or, more succintly
intput_shape = x_train.shape[1:4]
# Python stops before the last index of a range, so 1:4 => 1,2,3

# We finish adjusting our images by changing them to floating point
# and scaling them down to the range [0,1], which is preferable
# for deep learning applications
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# The final touch is to convert class vectors to binary class matrices
# The target MNIST data comes in the form of a class label, such as "3"
# Our convolutional model will need them to be in the form of "one-hot"
# vectors, such as [0,0,0,1,0,0,0,0,0,0]
# Luckily, Keras has a built in function for the conversion
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Note that this adds an axis to each of these arrays
# Our data is now all ready for training!

#%% Building a convolutional neural network model
# Now we will build our CNN that will classify these images
# Building a model in Keras is as simple as adding layers in order

# Start with a sequential model
model = Sequential()
# Add the first layer- a 2D convolution. For this very first layer,
# we will need to define an input shape. The rest of the layers will
# infer the shape based on the previous layer
# We will use 16 kernels (filters) with a standard size of (3,3) and the standard
# ReLU activation function
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# Now add a second layer, this time with more kernels
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
# At this point, most standard models use what's called Max Pooling
# which sub-samples the input. Instead, we will use a strided convolution
# which accomplishes the same subsampling but is faster and generally
# more effective
model.add(Conv2D(32, kernel_size=(3,3),
                 strides=(2,2),
                 activation='relu'))
# Dropout is a common regularization technique to prevent overfitting
model.add(Dropout(0.25))
# Now, flatten what's left into a single row so that a multi-layer perceptron
# can be added to the end of the model for classification
model.add(Flatten())
# Keras calls the standard neural network layer "Dense"
model.add(Dense(128, activation='relu'))
# Dropout is more important in MLPs than in CNNs
model.add(Dropout(0.5))
# This is our final layer. We need to output to neurons that correspond
# to each class. We also use the softmax activation, which effectively
# calculates probabilities of each class. This is the reason that
# we needed to reformat our target data previously
model.add(Dense(num_classes, activation='softmax'))

#%% Compiling and training the model

# Compiling the model is the final step before it is ready to train.
# We need to define our loss function, optimizer, and any additional
# metric that Keras will report as the training progresses
# Categorial cross-entropy is a standard loss for this type of classification
# The ADAM optimizer is widely used with good performance on the majority
# of deep learning applications
# Finally, we ask Keras to report the accuracy, since that we what we
# are really interested in
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# All that's left to do is to "fit" the model to our data!
# We supply our training data, our batch size and epochs that were
# defined earlier, we ask Keras to constantly report progress (verbose), 
# and we supply some validation data that will be evaluated at the end
# of every epoch so we can keep an eye on overfitting
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# After the training is complete, we evaluate the model again on our test
# data to see the results. This method will return a list of
# the loss value, categorical cross-entropy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Let's dislay some of our test images and see how the model does
# grab the first 20 images and display them in a panel
display_ims = np.rollaxis(x_test[:10,...,0],0,2).reshape(28,10*28)
fig0 = plt.figure(0)
plt.imshow(display_ims,cmap='gray')
# Now, get the predictions on those same images and print them out
# First, get the predictions. Remember that these will come in the form of
# probabilities for each class for each sample
probabilities = model.predict(x_test[:10])
# Now we find the maximum probability of each sample using a Numpy
# function called argmax, which returns indices where maximum accurs
predictions = np.argmax(probabilities,axis=1)
# Alternatively, we can have Keras do this for us with 
# the predict_classes method
predictions = model.predict_classes(x_test[:10])
# Now print these predictions in the console and see if they
# match up with the images!
print('Class predictions are: ',predictions)
# Depending on the random initialization of your model, it likely
# got 9/10 or 10/10 correct
# Neat!

# Part 1 is now complete! We can always get better results by training for more
# epochs, and with the full set of training data

#%% Part 2: Segmentation
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
# Now it's time to 'go back up': regain that lost spatial resolution
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




