# Use the nvcr.io/nvidia/pytorch:22.11-py3 base image
FROM baristimunha/moabb


COPY requirements.txt .
# Install the Python packages listed in the 'meta_requirements.txt' file
RUN pip3 install -r requirements.txt


