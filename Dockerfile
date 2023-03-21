FROM ubuntu:20.04

MAINTAINER Louis Liu <s041978@hotmail.com> 

WORKDIR /mnt/

ARG DEBIAN_FRONTEND=noninteractive

# Install package dependencies
RUN apt-get update &&\
    apt-get install -y --no-install-recommends software-properties-common build-essential \
            autoconf libtool pkg-config ca-certificates libssl-dev pkg-config libprotobuf-dev \
            git unzip wget vim man cppman \
            automake cmake g++ gcc \
            gdb valgrind && \
    apt-get clean

# Install Python dependencies
RUN apt-get update && \
    apt-get install -y libavcodec-dev libavformat-dev libswscale-dev \
            libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
            libgtk-3-dev libpng-dev libopenexr-dev libtiff-dev libwebp-dev \
            python3 python3-pip python3-dev python3-numpy python3-setuptools

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel

# Install Python module dependencies
RUN pip install --root-user-action=ignore  requests pandas Ipython tabulate xlwings matplotlib xlsxwriter

ENTRYPOINT /bin/bash
