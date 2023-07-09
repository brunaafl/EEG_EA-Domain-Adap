FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN useradd --uid 1006 -U --create-home --shell /bin/bash gabriel

WORKDIR /workspace/project
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN ["mkdir","-m777", "/.mne"]
RUN ["mkdir","-m777", "/workspace/outputs"]
RUN ["mkdir","-m777", "/workspace/datasets"]

