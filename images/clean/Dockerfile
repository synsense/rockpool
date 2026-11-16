# Start from a base image with cuda support (driver 580)
FROM nvidia/cuda:13.5.0-cudnn-devel-ubuntu26.04

# - Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda

# - Ensure we run non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# Configure the base environment
RUN apt-get clean && \
    apt-get update -y && \
    apt-get install --no-install-recommends -y \
        apt-transport-https \
        build-essential \
        mesa-common-dev \
        libgl1 \
        libglx-mesa0 \
        libegl1 \
        libopengl0 \
        libglu1-mesa-dev \
        libxrandr-dev \
        libxrandr2 \
        libxinerama-dev \
        libxcursor-dev \
        libxcursor1 \
        libxi-dev \
        libxi6 \
        libxcomposite1 \
        libxdamage1 \
        libxss1 \
        libxtst6 \
        libxcb-dri2-0-dev \
        libxcb-dri3-dev \
        libxcb-present-dev \
        libxcb-sync-dev \
        libx11-xcb-dev \
        libxcb-glx0-dev \
        libasound2t64	 \
        libxml2-16 \
        wget \
        git \
        ninja-build \
        pandoc \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13-full \
        python3.13-dev \
        python3.13-venv \
        python3.13-tk \
        python-is-python3 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 && \
    update-alternatives --set python3 /usr/bin/python3.13 && \
    rm -rf /var/lib/apt/lists/*

# Create an isolated Python environment for Rockpool dependencies
RUN python3 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools inside the virtual environment
RUN python -m pip install --upgrade pip setuptools wheel

# Copy the requirements file for Rockpool
COPY ./resources/requirements.txt ./requirements.txt

# Install Rockpool dependencies in the virtual environment
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt

ENV CUDA_HOME=/usr/local/cuda

# - Set default entry command
CMD ["/bin/bash", "--login"]
