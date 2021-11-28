FROM ubuntu:20.04

RUN apt-get update


# curl/wget/git
RUN apt-get install -y curl wget git
# vim/emacs
RUN apt-get install -y vim
# Shell
RUN apt-get install -y bc
RUN apt-get install -y shellcheck
# C
RUN apt-get install -y build-essential gcc
RUN apt-get install -y libc6-dev-i386
RUN apt-get install -y libssl-dev
RUN apt-get install -y nasm

# Pip
RUN apt-get install -y python3-pip
RUN pip3 install pycodestyle
RUN pip3 install pep8

# NumPy
RUN pip3 install numpy==1.19.2
RUN pip3 install psutil

# Ruby
RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y lsof

RUN apt-get install -y sudo

# Create test user
RUN useradd -M correction_tester


RUN pip install requests
RUN pip install tensorflow==2.6
RUN pip install matplotlib
RUN pip install numpy==1.19.2
RUN pip install pycodestyle==2.6.0
RUN pip install scipy==1.6.2
RUN pip install opencv-python
RUN pip install Pillow

RUN wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5
RUN wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
RUN mkdir -p ~/.keras/models
RUN mv vgg19_weights_tf_dim_ordering_tf_kernels*.h5 ~/.keras/models