# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:10.2-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        # libnccl-dev \
        && \
        rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# https://qiita.com/yagince/items/deba267f789604643bab
ENV DEBIAN_FRONTEND=noninteractive

# https://github.com/phusion/baseimage-docker/issues/319#issuecomment-262550835
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        mercurial \
        libffi-dev \
        libreadline-gplv2-dev \
        libncursesw5-dev \
        libssl-dev \
        libsqlite3-dev \
        tk-dev libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        liblzma-dev \
        libpng-dev && \
        rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# https://github.com/pyenv/pyenv-installer
RUN curl https://pyenv.run | bash
RUN pyenv install 3.7.4
RUN pyenv global 3.7.4

RUN rm -rf /usr/bin/python* /usr/bin/pip*

WORKDIR /code
COPY requirements.txt /code
RUN pip install -U pip
RUN pip install -r requirements.txt
# Install jupyterlab vim extension, which requires nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install nodejs
RUN jupyter labextension install @axlair/jupyterlab_vim
RUN jupyter labextension install @jupyterlab/toc
