# Start from a base image with cuda support (driver 535)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

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
    apt-get install -y python3.11 python3-pip && \
    apt-get install -y python-is-python3 && \
    apt-get install -y python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file for Rockpool
COPY ./resources/requirements.txt ./

# Update the python environment
RUN pip install -r requirements.txt && \
    rm -f requirements.txt

ENV CUDA_HOME=/usr/local/cuda

# - Set default entry command
CMD ["/bin/bash", "--login"]
