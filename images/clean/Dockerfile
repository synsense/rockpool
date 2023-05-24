# Start from a base image with cuda support (driver 525)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# - Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda

# - Ensure we run non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# Configure the base environment
RUN apt-get clean && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y apt-transport-https && \
    apt-get install --no-install-recommends -y build-essential libgl1-mesa-glx mesa-common-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libx11-xcb-dev libxcb-glx0-dev && \
    apt-get install --no-install-recommends -y libglu1-mesa-dev libxml2 libxdamage1 libopengl0 && \
    apt-get install --no-install-recommends -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 && \
    apt-get install --no-install-recommends -y wget git ninja-build && \
    apt-get install --no-install-recommends -y pandoc && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR $HOME/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init

# Copy the requirements file for Rockpool
COPY ./resources/clean.yml ./

# Update the python environment
RUN conda update --all && \
    conda env update --name base --file clean.yml && \
    rm -rf clean.yml

ENV CUDA_HOME=/miniconda

# - Set default entry command
CMD ["/bin/bash", "--login"]
