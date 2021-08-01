#Download base image ubuntu 20.04
FROM ubuntu:20.04

# Dockerfile info
LABEL maintainer="monte.igna@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image \
    with all the components required to run the scripts."

ARG PYTHON_VERSION=3.8
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update

# Install dependecies
RUN apt-get install -y \
    git \
    python3-pip \
    cmake \
    python3-psutil \
    python3-future  \
    python3-dev \
    libeigen3-dev \
    libcppunit-dev

RUN apt-get install -y --reinstall ca-certificates

RUN mkdir -p repos

WORKDIR /repos

# build kdl
RUN git clone https://github.com/orocos/orocos_kinematics_dynamics.git && \
    cd orocos_kinematics_dynamics && \
    git checkout release-1.5 && \
    cd orocos_kdl && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=release && \
    make -j && make install

COPY pykdl.patch orocos_kinematics_dynamics/

# build kdl python bindings
RUN cd orocos_kinematics_dynamics && \
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
    pip install opencv-python-headless

# install fork of ros-less kdl parser
RUN git clone https://github.com/imontesino/kdl_parser.git && \
    pip install -r kdl_parser/requirements.txt && \
    pip install kdl_parser/

ADD . /repos/pybullet-iiwa

WORKDIR /repos/pybullet-iiwa

RUN pip install -r requirements.txt
