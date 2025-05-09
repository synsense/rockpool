# Start from a base image with cuda support (driver 525)
FROM registry.gitlab.com/synsense/rockpool/clean:latest

# - Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda

# - Ensure we run non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Change shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# Copy the requirements file for Rockpool
COPY ./resources/requirements.txt ./

# Configure the rockpool requirements (base environment comes from clean docker image)
RUN pip install -r requirements.txt && \
    rm -f requirements.txt

# - Set default entry command
CMD ["/bin/bash", "--login"]
