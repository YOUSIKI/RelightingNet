FROM tensorflow/tensorflow:latest

RUN pip install -i https://mirrors.aliyun.com/pypi/simple \
    scikit-image \
    tensorflow-addons

WORKDIR /workspace
