#!/bin/bash
sudo apt update && sudo apt upgrade -y
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# Download the latest Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Install Miniconda silently
bash miniconda.sh -b -p $HOME/miniconda

# Remove the installer
rm miniconda.sh

# Initialize conda for bash shell
$HOME/miniconda/bin/conda init bash

# Activate installation
source $HOME/.bashrc

echo "Miniconda has been installed silently."