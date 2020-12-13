FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y build-essential software-properties-common \
    apt-transport-https wget gcc g++

# Install cuda toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

RUN apt-get update && apt-get -y install cuda-toolkit-10-1

ENV PATH /usr/local/cuda/bin:$PATH

# Install gpgpu-sim dependencies
RUN apt-get install -y build-essential xutils-dev \
    bison zlib1g-dev flex libglu1-mesa-dev doxygen graphviz git
    
ENV CUDA_INSTALL_PATH /usr/local/cuda

WORKDIR /home

# Clone and build gpgpu-sim
RUN git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
WORKDIR /home/gpgpu-sim_distribution/
# RUN git checkout 90ec3399763d7c8512cfe7dc193473086c38ca38
RUN . ./setup_environment && make
RUN mkdir /home/workdir && cd /home/workdir && \
    ln -s /home/gpgpu-sim_distribution/configs/tested-cfgs /home/workdir/configs

# Install text editor
RUN apt-get install -y nano vim

# Start from home directory
RUN cp -rT $HOME /home/workdir && \
    echo "source /home/gpgpu-sim_distribution/setup_environment" >> /home/workdir/.bashrc
ENV HOME /home/workdir
WORKDIR $HOME
