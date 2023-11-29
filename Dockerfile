# Set the base image to a suitable Linux distribution
FROM ubuntu:20.04

# Set noninteractive mode and set the timezone to America/Detroit
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Detroit

# Install required dependencies
RUN apt-get update && apt-get install -y curl bzip2

# Set the ANACONDA_ACCEPT_LICENSE environment variable to "yes"
ENV ANACONDA_ACCEPT_LICENSE="yes"

# Download and install Anaconda with automatic "yes" response
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 -u $ANACONDA_ACCEPT_LICENSE

# Add Conda binary directory to PATH
ENV PATH="/opt/anaconda3/bin:${PATH}"

# Create Conda environment and activate it
RUN conda create -y --name env36 python=3.6
ENV PATH="/opt/anaconda3/envs/env36/bin:${PATH}"

# Install PyTorch and other packages within the Conda environment
RUN /opt/anaconda3/envs/env36/bin/conda install -y pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia
RUN /opt/anaconda3/envs/env36/bin/conda install -c conda-forge -y onnx

# Install additional Python packages
RUN /opt/anaconda3/envs/env36/bin/pip install tqdm tensorboard future pydot
RUN /opt/anaconda3/envs/env36/bin/pip3 install -U scikit-learn scipy matplotlib

# Install other system dependencies
RUN apt-get update && apt-get install -y git graphviz numactl wget vim unzip
RUN git clone https://github.com/mlperf/logging.git mlperf-logging
RUN /opt/anaconda3/envs/env36/bin/pip install -e mlperf-logging

# Set the working directory
RUN mkdir /code
WORKDIR /code
