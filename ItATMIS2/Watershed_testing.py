import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
import sys
import timeit

try:
    sys.path.insert(1,'/home/jmj136/deep-learning/Utils')
    from VisTools import multi_slice_viewer0

    data_path = 'ItATMIS2/TestNiftis/ItATMISdata_lungs.h5'
except Exception as e:
    print(e)


try:
    with h5py.File(data_path,'r') as hf:
        images = np.array(hf.get('images'))
        volshape = images.shape
        inds= np.array(hf.get('inds'))
        file_list_temp = hf.get('file_list')
        file_list = [n[0].decode('utf-8') for n in file_list_temp]
        datadir_temp = hf.get('datadir')
        datadir = datadir_temp[0][0].decode('utf-8')
        if 'model_path' in list(hf.keys()):
            modelpath_temp = hf.get('model_path')
            model_path = modelpath_temp[0][0].decode('utf-8')
        if 'annotation_file' in list(hf.keys()):
            annotationfile_temp = hf.get('annotation_file')
            AnnotationFile = annotationfile_temp[0][0].decode('utf-8')
        FNind = np.array(hf.get('FNind'))
        targets = np.array(hf.get('targets'))
except Exception as e:
    print(e)

img = images[40,...]

gradients = np.zeros_like(images)
for ss in range(images.shape[0]):
    gradients[ss] = sobel(images[ss])

print(gradients.shape)

segments_watershed = watershed(np.rollaxis(gradients,0,3), markers=1000, compactness=0.0001)
print(segments_watershed.shape)
segments_watershed = np.rollaxis(segments_watershed,2,0)


multi_slice_viewer0(segments_watershed/np.max(segments_watershed))
plt.show()