# CUDA12 image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# ================ Install Python from source code =================
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    libbz2-dev \
    liblzma-dev \
    git

# Install and compile specific version (Python 3.11.3)
#WORKDIR /tmp
#RUN wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz
#RUN tar -xvf Python-3.11.3.tgz
#WORKDIR /tmp/Python-3.11.3
#RUN ./configure --enable-optimizations
#RUN make altinstall

# Cleanup
#WORKDIR /
#RUN rm -r /tmp/Python-3.11.3
#RUN rm /tmp/Python-3.11.3.tgz

# Set python3.11 as default python3
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1

# Install and compile specific version (Python 3.10.15), since NVFlare is not compatible with Python 3.11
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.15/Python-3.10.15.tgz
RUN tar -xvf Python-3.10.15.tgz
WORKDIR /tmp/Python-3.10.15
RUN ./configure --enable-optimizations
RUN make altinstall

# Cleanup
WORKDIR /
RUN rm -r /tmp/Python-3.10.15
RUN rm /tmp/Python-3.10.15.tgz


# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install huggingface_hub package
RUN pip install huggingface_hub

# Define build argument for HF_TOKEN
ARG HF_TOKEN

# Set environment variable during build
ENV HF_TOKEN=${HF_TOKEN}

# Copy the rest of the application
COPY . .

# Expose the port (optional, based on your application)
EXPOSE 5000/tcp
# Set the CMD to use accelerate launch
CMD ["python3", "main.py", "--conf", "templates/llm_finetuning/configs/conf.yaml"]