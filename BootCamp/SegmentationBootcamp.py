'''Code demonstration for introductory deep learning programming
in Keras. This tutorial walks through an example in segmentation
'''
# for pydicom install: conda install -c conda-forge --no-deps pydicom
#%% Part 1: Classification
#%% Initial Preparation
# First, import necessary modules
import os # operating system operations 
import numpy as np # number crunching
import keras # our deep learning library
import matplotlib.pyplot as plt # for plotting our results

#%% Data preparation
# As with all deep learning applications, we need to do some data
# preparation first. It's generally far more time consuming than the actual
# deep learning part.
# This guide assumes that you have the data downloaded and extracted to
# the sub-directory 'LCTSC' in your working directory

# We have to come up with a way to load in all that data in an organized way.
# First, let's get all the subject directories. We'll do this by proceeding
# through the directory structure and grabbing the ones we want.
# We'll use the package glob to make this easy
import glob
# We know our initial directory: LCTSC. Let's add that to our current
# directory to get the full path
initial_dir = os.path.join(os.getcwd(),'LCTSC')
# Now we'll get all the subject directories using glob
subj_dirs = glob.glob(os.path.join(initial_dir,'LCTSC*'))
# Now all the subject directories are contained in a list
# Let's grab the first one in that list and look for the data
cur_dir = subj_dirs[1]
# The next directory level just has 1 directory, so we'll grab that
cur_dir = glob.glob(os.path.join(cur_dir, "*", ""))[0]
# Now we have the dicom image directory and the label directory
# The dicom iamge directory starts with a 0 so we'll find that one first
dcm_dir = glob.glob(os.path.join(cur_dir, "0*", ""))[0]
# Let's grab the label directory while we're at it. It starts with a 1
lbl_dir = glob.glob(os.path.join(cur_dir, "1*", ""))[0]
# Now, we can get the list of dicom files that we need to load for this subject
# We just have to look for .dcm files in the dcm_dir we found
dicom_files = glob.glob(os.path.join(dcm_dir, "*.dcm"))
# Great. Let's get the label filepath too
# It's just contained in a single dicom-rt file in the label directory
lbl_file = glob.glob(os.path.join(lbl_dir,"*.dcm"))[0]
# Great! We have all the file paths for this subject. Now we need
# to actually load in the data
# We'll need the PyDicom package to read the dicoms
import pydicom
# First, we'll load in all the dicom data to a list
dicms = [pydicom.read_file(fn) for fn in dicom_files]
# These likely won't be in slice order, so we'll need to sort them
dicms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
# Then, stack all the pixel data together into a 3D array
# We'll convert to floats while doing this
ims = np.stack([dcm.pixel_array.astype(np.float) for dcm in dicms])
# The last thing we will do is normalize all the images to [0,1]
# There are a variety of normalization methods used, but
# this is simple and seems to work just fine
for im in ims:
    im /= np.max(im)
    
# Now that we have our inputs, we need targets for our
# deep learning model.
# Let's go back and load the label file we already found
label = pydicom.read_file(lbl_file)
# This gets ugly, but we need to extract the contour data
# stored in the label and convert it to masks that can be fed
# to the deep learning model.
# First, get the contour data. We will focus on the lungs for this tutorial
# We need to figure out which contours are the lungs
contour_names = [s.ROIName for s in label.StructureSetROISequence]
# Get the right and left lung indices
r_ind = contour_names.index('Lung_R')
l_ind = contour_names.index('Lung_L')
# Extract the corresponding contours and combine
contour_right = [s.ContourData for s in label.ROIContourSequence[r_ind].ContourSequence]
contour_left = [s.ContourData for s in label.ROIContourSequence[l_ind].ContourSequence]
contours = contour_left + contour_right
# Next, we need to setup the coordinate system for our images
# to make sure our contours are aligned
# First, the z position
z = [d.ImagePositionPatient[2] for d in dicms]
# Now the rows and columns
# We need both the position of the origin and the
# spacing between voxels
pos_r = dicms[0].ImagePositionPatient[1]
spacing_r = dicms[0].PixelSpacing[1]
pos_c = dicms[0].ImagePositionPatient[0]
spacing_c = dicms[0].PixelSpacing[0]
# Now we are ready to create our mask
# First, preallocate
mask = np.zeros_like(ims)    
# we are going to need a contour-to-mask converter
from skimage.draw import polygon
# now loop over the different slices that each contour is on
for c in contours:
    nodes = np.array(c).reshape((-1, 3))
    assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
    zNew = [round(elem,1) for elem in z]
    try:
        z_index = z.index(nodes[0,2])
    except ValueError:
        z_index = zNew.index(nodes[0,2])
    r = (nodes[:, 1] - pos_r) / spacing_r
    c = (nodes[:, 0] - pos_c) / spacing_c
    rr, cc = polygon(r, c)
    mask[z_index,rr, cc] = 1

# Now we have a mask!
# We have all the pieces we need:
# Inputs, and targets
# Now we just to repeat for all of the subjects
# Luckily, there is a pre-made function that does everything we
# just did for whatever directory we give it. So all we have to do is
# call this function on all the directories we already collected
from Demo_Functions import GetLCTSCdata
data = [GetLCTSCdata(d) for d in subj_dirs]
# get all images together as inputs
inputs = np.concatenate([d[0] for d in data])
# get all masks together as targets
targets = np.concatenate([d[1] for d in data])
# clear a couple large variables that are no longer needed
del data
del ims

# Just a couple more pre-processing steps.
# First, our images are 512x512. That's pretty large for most
# deep learning applications. It's certainly doable, but for the
# purpose of this demonstration we will downsample to
# 256x256 so that the processing is faster
# we'll use another scipy function for this
from scipy.ndimage import zoom
inputs = zoom(inputs, (1,.5,.5))
targets = zoom(targets, (1,.5,.5))
# the final step is to add a singleton dimesion to these arrays
# This is necessary because the deep learning model we will create
# will expect our input to have color channels. Since our images
# are grayscale, they will just have a single color channel
inputs = inputs[...,np.newaxis]
targets = targets[...,np.newaxis]

# So far so good! Our data is now ready for training!

# But, wait. We don't just need training data. We need a 
# way of determining if our model is overfitting
# We can split some of our data off and use it for validation
# during the training process
# Let's take 20% of the last slices and use them for this purpose
# This will be equal to the last two subjects, so we won't
# have any overlap of subjects between the different sets
# Get the total number of slices
num_slices = inputs.shape[0]
# Find the cutoff
split_ind = np.int(.8*num_slices)
# split into training and validation using common nomenclature
x_train = inputs[:split_ind]
y_train = targets[:split_ind]
x_val = inputs[split_ind:]
y_val = targets[split_ind:]
# clear up unneeded variables
del inputs,targets

#%% Building a segmentation network

# We will build a deep convolutional neural network layer by layer
# We first need an input layer that takes our inputs
from keras.layers import Input
# Our input layer just needs the shape of the input we are providing
# The shape dimensions are sample,row,column,channel
# For this 2D network, our samples are different slices
# We don't need to provide this dimension to the input layer, since
# we will feed those samples in as batches during training. But
# we need the rest of the dimensions
inp = Input(shape=x_train.shape[1:])
# Now, we will add on convolutional layers
from keras.layers import Conv2D
# We can reuse the variable 'x' and Keras will remember what the layers
# are connected to
x = Conv2D(8,(3,3),activation='relu')(inp)
x = Conv2D(16,(3,3),activation='relu')(x)
x = Conv2D(16,(3,3),activation='relu')(x)
# now we will use a strided convolution, which downsamples the input
# and increases the network's receptive field
# We will use zero padding first to make the image shapes work out correctly
from keras.layers import ZeroPadding2D
x = ZeroPadding2D(padding=(1,1))(x)
x = Conv2D(16,(4,4),strides=(2,2),activation='relu')(x)
# repeat that sequence
x = Conv2D(32,(3,3),activation='relu')(x)
x = Conv2D(64,(3,3),activation='relu')(x)
x = Conv2D(64,(3,3),activation='relu')(x)
x = ZeroPadding2D(padding=(1,1))(x)
x = Conv2D(64,(4,4),strides=(2,2),activation='relu')(x)
# now, we will reverse the downsampling using transposed convolutions
from keras.layers import Conv2DTranspose
x = Conv2DTranspose(64,(4,4),strides=(2,2),activation='relu')(x)
x = Conv2DTranspose(32,(3,3),activation='relu')(x)
x = Conv2DTranspose(32,(3,3),activation='relu')(x)
x = Conv2DTranspose(32,(4,4),strides=(2,2),activation='relu')(x)
x = Conv2DTranspose(16,(3,3),activation='relu')(x)
x = Conv2DTranspose(16,(3,3),activation='relu')(x)
x = Conv2DTranspose(16,(3,3),activation='relu')(x)
# finally, our output layer will need to have a single output
# channel corresponding to a single segmentation class
# We will use sigmoid activation that squashed the output to a probability
out = Conv2D(1,(1,1),activation='sigmoid')(x)
# Now, we have a graph of layers created but they are not yet a model
# Fortunately, Keras makes it easy to make a model out of a graph
# just using the input and output layers
from keras.models import Model
SegModel = Model(inp,out)
# We have a deep learning model created! Let's take a look to make
# sure we got the image shapes to work out
SegModel.summary()

#%% Compiling and training the model

# Compiling the model is the final step before it is ready to train.
# We need to define our loss function and optimizer that Keras will 
# use to run the training

# The Dice coefficient is not only a good segmentation metric,
# is also works well as a segmentation loss function since it
# can be converted to being differentiable without much difficulty
# Loss functions in Keras need be defined using tensor functions
# Here is what that looks like:
import keras.backend as K
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)
    # We have calculated dice, but we want to maximize it. Keras tries to minimize
    # the loss so we simply return 1- dice
    return 1-dice


# The ADAM optimizer is widely used with good performance on the majority
# of deep learning applications
SegModel.compile(loss=dice_coef,optimizer=keras.optimizers.Adam())

# All that's left to do is to "fit" the model to our data!
# We supply our training data, our batch size and epochs that were
# defined earlier, we ask Keras to constantly report progress (verbose), 
# and we supply some validation data that will be evaluated at the end
# of every epoch so we can keep an eye on overfitting
SegModel.fit(x_train, y_train,
          batch_size=16,
          epochs=2,
          verbose=1,
          shuffle=True,
          validation_data=(x_val, y_val))
# After the training is complete, we evaluate the model again on our validation
# data to see the results.
score = SegModel.evaluate(x_val, y_val, verbose=0)
print('Final Dice:', 1-score[0])


# We'll display the prediction and truth next to each other
# and see how it faired
predictions = SegModel.predict(x_val)
fig5 = plt.figure(5)

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
# So far, we've been making sequential models.
# Basically, it means that our network
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

inp = Input(shape=x_train.shape[1:])

# Right now, 'inp' defines our input layer. In the 
# next step, we will provide 'inp' as an argument to our
# next layer, which will then be called 'x1'. We need
# to keep track of these now since we will be using the layers
# a second time later
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
func_model.fit(x_train, y_train,
          batch_size=16,
          epochs=2,
          verbose=1,
          validation_data=(x_val, y_val))

predictionsF = SegModel.predict(x_val)
fig6= plt.figure(6)


# Well.... ok. It's about the same.
# However! In the long run (more than 2 epochs), having these skip connections
# will definitely make a difference. The difference becomes more pronounced
# for deeper networks (more layers) with more parameters and larger images.

# Now that you know the functional API, you can make any graph you like, train
# it, and use it! Once you've mastered the syntax and conceptual understanding
# of how to connect layers, you are only limited by your imagination as far
# as what kind of network you can build.

# Best of luck, and happy deep learning!