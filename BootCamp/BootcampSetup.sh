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
# Download cuDNN library
echo "Downloading cuDNN library"
gsutil cp gs://ml4mi_bootcamp/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb
# install cuDNN library
sudo dpkg -i libcudnn7_7.1.4.18-1+cuda9.0+amd64.deb

# install Anaconda
echo "Installing Anaconda"
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b -p ~/anaconda3
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH
conda create -y --name env_keras python=3.5
echo 'source activate env_keras' >> ~/.bashrc
source activate env_keras
# configure jupyter notebook
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
sudo ufw allow 8888/tcp
# install other packages
pip install -q tensorflow-gpu
pip install -q keras
conda install -y scikit-image
conda install -y scipy
conda install -cy conda-forge --no-deps pydicom
# download jupyter notebook and scripts
wget https://github.com/jmj23/deep-learning/raw/master/BootCamp/SegmentationBootcamp.ipynb
wget https://github.com/jmj23/deep-learning/raw/master/BootCamp/Demo_Functions.py
# download segmentation data
echo "Downloading LCTSC data"
gsutil -q cp -r gs://ml4mi_bootcamp/LCTSC .
# download classification data
# first install unzip
sudo apt-get isntall unzip -y
echo "Downloading NIH CXR data"
gsutil -q cp gs://ml4mi_bootcamp/CXR_data.zip .
gsutil -q cp gs://ml4mi_bootcamp/male_female_basic_example.ipynb .
unzip data.zip

# setup jupyter notebook password
echo "Setting up jupyter notebook password"
jupyter notebook password

echo "Reboot is required"
