ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV PYTHONPATH=.
ENV FORCE_CUDA="1"
ENV MKL_SERVICE_FORCE_INTEL="1"
ENV MKL_THREADING_LAYER="GNU"

LABEL pytorch="1.8.0"
LABEL cuda="11.1"
LABEL cudnn="8"
LABEL ymir="1.1.0"


RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# To fix GPG key error when running apt-get update
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt install -y gnupg2 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update && apt install -y git vim libgl1-mesa-glx ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && rm -rf /var/lib/apt/lists/* && \
mkdir -p /img-man && echo "python3 ymir/start.py" > /usr/bin/start.sh

COPY ./requirements.txt /workspace
RUN pip install -r /workspace/requirements.txt

COPY . /app
WORKDIR /app

RUN cd /app/ops && bash make.sh && \
    mv /app/ymir/img-man/*.yaml /img-man && \
    pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.3.0"

CMD bash /usr/bin/start.sh
