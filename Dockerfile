# Set the base image to a suitable Linux distribution
FROM ubuntu:20.04

# Install required dependencies
RUN apt-get update && apt-get install -y curl bzip2

# Set the ANACONDA_ACCEPT_LICENSE environment variable to "yes"
ENV ANACONDA_ACCEPT_LICENSE="yes"

# Download and install Anaconda with automatic "yes" response
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh && \
    bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 -u $ANACONDA_ACCEPT_LICENSE

# Set the working directory
WORKDIR /code
ADD . .
# You can now activate the Conda environment and use it within the container
# CMD ["conda", "activate", "env36"]
