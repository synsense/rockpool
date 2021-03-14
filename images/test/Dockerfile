# Start from a base image including conda
FROM continuumio/miniconda3:latest

# Copy the requirements file for Rockpool
COPY ../requirements.yml .

# Configure the environment
RUN [ "conda", "update", "--all" ]
RUN [ "conda", "env", "update", "--name", "base", "--file", "requirements.yml" ]
RUN [ "pip", "install", "--upgrade", "jax", "jaxlib", "sphinx-rtd-theme", "sphinx-autobuild" ]
