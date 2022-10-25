ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV PYTHONPATH=.
ENV FORCE_CUDA="1"
ENV MKL_SERVICE_FORCE_INTEL="1"
ENV MKL_THREADING_LAYER="GNU"

LABEL pytorch="1.6.0"
LABEL cuda="10.1"
LABEL cudnn="7"
LABEL ymir="1.1.0"

# To fix GPG key error when running apt-get update
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
apt update && apt install -y git vim libgl1-mesa-glx ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && rm -rf /var/lib/apt/lists/* && \
mkdir -p /img-man && echo "python3 ymir/start.py" > /usr/bin/start.sh

COPY . /app
WORKDIR /app

RUN cd /app/ops && bash make.sh && \
    cd /app && pip install -r requirements.txt && \
    mv /app/ymir/img-man/*.yaml /img-man && \
    pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.3.0"

CMD bash /usr/bin/start.sh
