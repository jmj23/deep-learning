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

# install Anaconda
echo "Installing Anaconda"
wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh
bash Anaconda3-5.3.0-Linux-x86_64.sh -b -p ~/anaconda3
echo 'export PATH=~/anaconda3/bin:$PATH' >> ~/.bashrc
export PATH=~/anaconda3/bin:$PATH
conda create -y --name env_keras python=3.6
echo 'source activate env_keras' >> ~/.bashrc
source activate env_keras
# install and configure jupyter notebook
conda install -y notebook=5.6
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
sudo ufw allow 8888/tcp
# install other packages
conda install -y cython
conda install -y tensorflow-gpu
pip install -q keras
conda install -y scikit-image
conda install -y natsort
pip install -q nibabel
pip install -q pydicom


# setup jupyter notebook password
echo "Setting up jupyter notebook password"
jupyter notebook password

echo "Reboot is required"