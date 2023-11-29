# Set the base image to a suitable Linux distribution
FROM ubuntu:20.04

# Install required dependencies
RUN apt-get update && apt-get install -y curl bzip2

# Set the ANACONDA_ACCEPT_LICENSE environment variable to "yes"
ENV ANACONDA_ACCEPT_LICENSE="yes"

# Download and install Anaconda with automatic "yes" response
RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 -u $ANACONDA_ACCEPT_LICENSE
RUN conda create -y --name env36 python=3.6
RUN conda activate env36
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

RUN pip install tqdm tensorboard future pydot
RUN pip3 install -U scikit-learn scipy matplotlib

RUN apt-get update
RUN apt install -y git graphviz numactl wget vim unzip
RUN git clone https://github.com/mlperf/logging.git mlperf-logging
RUN pip install -e mlperf-logging

RUN conda install -c conda-forge -y onnx


# Set the working directory
RUN mkdir /code
WORKDIR /code
# You can now activate the Conda environment and use it within the container
# CMD ["conda", "activate", "env36"]
