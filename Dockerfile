FROM tensorflow/tensorflow:1.12.0-py3

RUN pip install -i https://mirrors.aliyun.com/pypi/simple \
    Pillow \
    scikit-image \
    opencv-python

WORKDIR /workspace
