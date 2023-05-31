FROM nvcr.io/nvidia/pytorch:23.05-py3

COPY requirements.txt .
RUN echo "teste"
RUN pip3 install -r requirements.txt


