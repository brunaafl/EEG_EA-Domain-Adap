FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace/project
COPY requirements.txt .
RUN pip3 install -r requirements.txt
