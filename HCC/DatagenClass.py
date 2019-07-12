# pylint: disable=no-member
"Doctring"
import cv2
import keras
import numpy as np

try:
    import scipy
    from scipy import linalg
    from scipy import ndimage
except ImportError:
    scipy = None


class NumpyDataGenerator(keras.utils.Sequence):
    "Image Data Generator with augmentation"
    "to be used for classification problems"
    "using .npy arrays as inputs"

    def __init__(self,
                 file_list,
                 labels,
                 batch_size=32,
                 dim=(256, 256),
                 n_channels=1,
                 shuffle=True,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 interpolation_order=1,
                 dtype='float32'):
        """initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = file_list
        self.n_channels = n_channels
        self.shuffle = shuffle

        # augmentation parameters
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        self.interpolation_order = interpolation_order

        # designate axes
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2
        
        # parse zoom parameter
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

        self.on_epoch_end()

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        y = np.empty((self.batch_size), dtype=np.float)
        # Generate data
        for i, f in enumerate(list_files_temp):
            # load and resize image
            im = np.rollaxis(np.load(f),0,3)
            if im.shape[:2] != self.dim:
                im_r = np.zeros(self.dim + (self.n_channels,))
                for c in range(self.n_channels):
                    im_r[...,c] = cv2.resize(im[...,c], self.dim)
                im = im_r

            # apply random transformation
            x = self.random_transform(im)

            # store image sample
            X[i, ] = x

            # store class label
            y[i] = self.labels[f]
        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        # edit this later after augmentation is implemented
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.
        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.
        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical}

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.
        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = self.apply_affine_transform(x, transform_parameters.get('theta', 0),
                                        transform_parameters.get('tx', 0),
                                        transform_parameters.get('ty', 0),
                                        transform_parameters.get('shear', 0),
                                        transform_parameters.get('zx', 1),
                                        transform_parameters.get('zy', 1),
                                        row_axis=img_row_axis,
                                        col_axis=img_col_axis,
                                        channel_axis=img_channel_axis,
                                        fill_mode=self.fill_mode,
                                        cval=self.cval,
                                        order=self.interpolation_order)

        if transform_parameters.get('flip_horizontal', False):
            x = self.flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = self.flip_axis(x, img_row_axis)

        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.
        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)

    # Axis flip- horizontal or vertical
    def flip_axis(self, x, axis):
        """ Performs a vertical or horizontal axis flip
        """
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def apply_affine_transform(self, x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                               row_axis=0, col_axis=1, channel_axis=2,
                               fill_mode='nearest', cval=0., order=1):
        """Applies an affine transformation specified by the parameters given.
        # Arguments
            x: 2D numpy array, single image.
            theta: Rotation angle in degrees.
            tx: Width shift.
            ty: Heigh shift.
            shear: Shear angle in degrees.
            zx: Zoom in x direction.
            zy: Zoom in y direction
            row_axis: Index of axis for rows in the input image.
            col_axis: Index of axis for columns in the input image.
            channel_axis: Index of axis for channels in the input image.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
            order int: order of interpolation
        # Returns
            The transformed version of the input.
        """
        if scipy is None:
            raise ImportError('Image transformations require SciPy. '
                              'Install SciPy.')
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[row_axis], x.shape[col_axis]
            transform_matrix = self.transform_matrix_offset_center(
                transform_matrix, h, w)
            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]

            channel_images = [ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
        return x
