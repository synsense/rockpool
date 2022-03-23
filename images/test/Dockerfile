# Start from a base image including conda
#FROM continuumio/miniconda3:latest
FROM nvidia/cuda:11.5.1-devel-ubuntu20.04

# - Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda

# - Ensure we run non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Change shell to bash
SHELL ["/bin/bash", "--login", "-c"]
#RUN ln -sf /bin/bash /bin/sh

# Copy the requirements file for Rockpool
COPY base.yml ./
COPY extras.yml ./

# Configure the base environment
RUN apt-get clean
RUN apt-get update -y
RUN apt-get install -y apt-transport-https
RUN apt-get install -y build-essential libgl1-mesa-glx mesa-common-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libx11-xcb-dev libxcb-glx0-dev
RUN apt-get install -y libglu1-mesa-dev libxml2 libxdamage1 libopengl0
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
RUN apt-get install -y wget git ninja-build
RUN rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR $HOME/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $CONDA_DIR
RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init


# Update the python environment
RUN conda update --all
RUN conda env update --name base --file base.yml
RUN pip install --upgrade jax jaxlib sphinx-rtd-theme sphinx-autobuild

ENV CUDA_HOME=/usr/local/cuda-11.5

# - Set default entry command
CMD ["/bin/bash", "--login"]
