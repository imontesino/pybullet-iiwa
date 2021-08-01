#Download base image ubuntu 20.04
FROM ubuntu:20.04

# Dockerfile info
LABEL maintainer="monte.igna@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image \
    with all the components required to run the scripts."

ARG PYTHON_VERSION=3.7
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    git

RUN apt-get install -y --reinstall ca-certificates

RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.7 \
    python3-pip

RUN mkdir -p repos

WORKDIR /repos

RUN git clone https://github.com/imontesino/pybullet-iiwa

WORKDIR /repos/pybullet-iiwa

RUN git branch -a

RUN git checkout  feature/add_basic_docs

RUN pip install -r requirements.txt

RUN python3 rl_tests.py

RUN cd ${CODE_DIR}/stable-baselines3 3&& \
    # required for SAC in parallel
    git checkout feat/multienv-off-policy && \
    pip install -e . && \
    # Use headless version for docker
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    rm -rf $HOME/.cache/pip

COPY . .

RUN yarn install --production
