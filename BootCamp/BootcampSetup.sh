#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-9-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  sudo dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda-9-0 -y
fi
# Enable persistence mode
sudo nvidia-smi -pm 1
# disable autoboost for performance
sudo nvidia-smi --auto-boost-default=DISABLED
# # make sure cuDNN tar file is present before these steps
# tar -xzvf cudnn-9.0-linux-x64-v7.tgz
# sudo cp cuda/include/cudnn.h /usr/local/cuda/include
# sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
# Make sure cuDNN .deb file is present before these steps
sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.0+amd64.deb

# install Anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc
conda env update
# configure jupyter notebook
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
sudo ufw allow 8888/tcp
# install other packages
pip install tensorflow-gpu
pip install keras
conda install scikit-image
conda install scipy
conda install -c conda-forge --no-deps pydicom
# download jupyter notebook and scripts
wget https://github.com/jmj23/deep-learning/raw/master/BootCamp/SegmentationBootcamp.ipynb
wget https://github.com/jmj23/deep-learning/raw/master/BootCamp/Demo_Functions.py
