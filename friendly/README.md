## Friendly Repository

This directory contains some deep learning coding scripts that are formatted in a way as to be
readable and accessible by others. It also contains a guide in the form of a PDF that can help users get
started with deep learning code.

### GettingStartedDL.pdf

If you arrived at this link hoping to get started with deep learning, this is where you should start. 
Download this pdf and follow the installation instructions to get the necessary software for coding
deep learning models. The second half of the guide walks through code which is available in this repository.

### CodingDemonstration.py

This is the code referenced in the previous guide. This is well-commented code that explains some
operations in Python, Numpy, and Keras that are necessary for deep learning coding. If you are trying
to write your first deep learning model, this script provides 2 basic examples using the popular
deep learning library Keras, which simplifies the task of building and using deep learning models.

The script contains a classification and segmentation example.

### PrepareSegmentationData.py

This script is meant as a demonstration of how data that is stored in the form of .mat files (from MATLAB)
can be prepared and formatted in preparation for training a deep learning model. This script pairs nicely
with the following.

### SegmentationTraining.py

This script assumes you formatted data for training a segmentation model akin to the above mentioned script.
This script loads in said data, creates a deep learning segmentation model (that can be tuned to your needs), 
trains it and shows the results.


## Happy deep learning!

Contact jmjohnson33@wisc.edu for questions, issues, or concerns with files in this repository if you
do not wish to use the github issues tool.



## License

The MIT License

Copyright (c) 2010-2018 Google, Inc. http://angularjs.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of the software contained in this repository and associated documentation files
(the "Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
