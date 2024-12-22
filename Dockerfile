ARG WORK_DIR=none 
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN apt update && apt install -y less nano git  

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    black \
    flake8 \
    h5py \ 
    flit \
    isort \
    jupyter \
    jupyterlab \
    lit \
    numpy==1.26.4 \
    pandas \
    Pillow \ 
    pre-commit \
    protobuf \ 
    shap \ 
    sentencepiece \
    scipy \ 
    scikit-learn \
    torchmetrics==0.10.3 \
    transformers \
    transformers-interpret \
    typing_extensions==4.7.1 

ENV FLIT_ROOT_INSTALL=1

RUN git clone https://github.com/HoustonJ2013/pytorch-frame && cd pytorch-frame && flit install 

RUN git config --global --add safe.directory /app
