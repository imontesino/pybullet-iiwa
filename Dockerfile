#Download base image ubuntu 20.04 with cuda 11.1 support
FROM nvidia/cudagl:11.1.1-base-ubuntu20.04

# Dockerfile info
LABEL maintainer="monte.igna@gmail.com"
LABEL version="0.1"
LABEL description="Docker image for RL on th KUKA LBR IIWA"

ARG PYTHON_VERSION=3.8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update

# Required for higher versions of cmake
RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget && \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update

# Install dependecies
RUN apt-get install -y --fix-missing \
    git \
    python3-pip \
    cmake \
    python3-psutil \
    python3-future  \
    python3-dev \
    libeigen3-dev \
    libboost-all-dev \
    libcppunit-dev \
    curl \
    gdb

# PyTorch with GPU support
RUN pip3 install \
    torch==1.8.2+cu111 \
    torchvision==0.9.2+cu111 \
    torchaudio==0.8.2 \
    -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

RUN  cmake --version

RUN mkdir -p repos

WORKDIR /repos

# build kdl
RUN git clone https://github.com/orocos/orocos_kinematics_dynamics.git && \
    cd orocos_kinematics_dynamics && \
    git checkout db25b7e480e068df068232064f2443b8d52a83c7 && \
    cd orocos_kdl && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=release && \
    make -j && make install

COPY pykdl.patch orocos_kinematics_dynamics/

# build kdl python bindings
RUN cd orocos_kinematics_dynamics && \
    git checkout db25b7e480e068df068232064f2443b8d52a83c7 && \
    git apply pykdl.patch && \
    git submodule update --init && \
    cd python_orocos_kdl && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=release && \
    make -j && make install && \
    export LD_LcdIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib && \
    ldconfig

# Use specific stable baselines version
RUN git clone https://github.com/DLR-RM/stable-baselines3 && \
    cd stable-baselines3 && \
    # required for SAC in parallel
    git checkout feat/multienv-off-policy && \
    pip install -e . && \
    # Use headless version for docker
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    # for training visualization
    pip install tensorboard

# install fork of ros-less kdl parser
RUN git clone https://github.com/imontesino/kdl_parser.git && \
    pip install -r kdl_parser/requirements.txt && \
    pip install kdl_parser/

# keys to download fri-client libraries
ARG FRI_CLIENT_REPO
ARG REPO_TOKEN

# build FRI-Client
RUN git clone https://${REPO_TOKEN}@${FRI_CLIENT_REPO} FRI_SDK && \
    cd FRI_SDK && \
    mkdir build && cd build && cmake .. && make -j && make install

# install the gym envs
RUN git clone https://${REPO_TOKEN}@github.com/imontesino/iiwa-fri-gym && \
    cd iiwa-fri-gym && \
    # Clone pybind11 source
    git submodule init && git submodule update && \
    # Build FRI python bindings
    mkdir build && cd build && cmake .. && make -j && make install && \
    # Install the python package
    cd .. && pip install . && \
    # Required by pybullet to output through the GUI
    apt-get update && apt-get install -y xvfb

# Required by pybullet to output through the GUI
RUN apt-get update && apt-get install -y xvfb
#libnvidia-gl-480

SHELL ["/bin/bash", "-c"]

RUN curl \
    -L https://raw.githubusercontent.com/docker/compose/1.29.2/contrib/completion/bash/docker-compose \
    -o /etc/bash_completion.d/docker-compose

