FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV CUDA_HOME=/usr/local/cuda

COPY . /app
WORKDIR /app

RUN <<EOF
sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
apt update && apt install -y git
cd /app/ops && bash make.sh
cd /app && pip install -r requirements.txt
EOF
