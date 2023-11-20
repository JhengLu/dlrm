# Set the base image to a suitable Linux distribution
FROM ubuntu:20.04

# Install required dependencies
RUN apt-get update && apt-get install -y curl bzip2

# Set the ANACONDA_ACCEPT_LICENSE environment variable to "yes"
ENV ANACONDA_ACCEPT_LICENSE="yes"

# Download and install Anaconda with automatic "yes" response
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 -u $ANACONDA_ACCEPT_LICENSE

WORKDIR tmp
ADD . .
# Add Conda to the PATH
ENV PATH="/opt/anaconda3/bin:$PATH"

# Create a Conda environment and install packages
RUN conda clean --all
RUN conda create --name env36 python=3.6

# Set the default command to execute the entry point script
CMD ["/entrypoint.sh"]

# Set the default environment
# SHELL ["conda", "run", "-n", "env36", "/bin/bash", "-c"]

# Install packages within the Conda environment
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    conda install -c conda-forge onnx -y

# Additional setup and package installations can go here

# Set the working directory
WORKDIR /code
ADD . .
# You can now activate the Conda environment and use it within the container
# CMD ["conda", "activate", "env36"]
