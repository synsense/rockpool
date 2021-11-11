# Start from a base image including conda
FROM continuumio/miniconda3:latest

# Copy the requirements file for Rockpool
COPY base.yml ./
COPY extras.yml ./

# Configure the environment
RUN conda update --all
RUN conda env update --name base --file base.yml
RUN conda env update --name base --file extras.yml
RUN pip install --upgrade jax jaxlib sphinx-rtd-theme sphinx-autobuild
RUN apt-get clean
RUN apt-get update -y
RUN apt-get install -y apt-transport-https
RUN apt-get install -y build-essential libgl1-mesa-glx mesa-common-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxcb-dri2-0-dev libxcb-dri3-dev libxcb-present-dev libxcb-sync-dev libx11-xcb-dev libxcb-glx0-dev
RUN apt-get install -y libglu1-mesa-dev
RUN rm -rf /var/lib/apt/lists/*

# libglu1mesa-dev
