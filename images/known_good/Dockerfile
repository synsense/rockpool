# Start from a base image with cuda support (driver 525)
FROM registry.gitlab.com/synsense/rockpool/clean:latest

# Copy the requirements file for Rockpool
COPY ./resources/known_good.yml ./

# Update the python environment
RUN conda update --all && \
    conda env update --name base --file known_good.yml && \
    rm -rf known_good.yml

# - Set default entry command
CMD ["/bin/bash", "--login"]
