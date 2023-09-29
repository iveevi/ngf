FROM ubuntu:22.04
WORKDIR /home/
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    vim git wget
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade matplotlib meshio pymeshlab seaborn torch torchvision tqdm
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
	&& mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
	&& wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb \
	&& dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb \
	&& cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt update && apt install -y cuda-toolkit-11-7 cuda-nvcc-11-7
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
