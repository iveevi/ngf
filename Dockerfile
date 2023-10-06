FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
WORKDIR /home/
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    libgl1 build-essential git ninja-build unzip vim wget
RUN pip3 install setuptools wheel imageio matplotlib largesteps meshio pymeshlab seaborn torch==2.0.1 torchvision==0.15.2 tqdm
RUN taskset --cpu-list 0-3 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN taskset --cpu-list 0-3 pip install "git+https://github.com/NVlabs/nvdiffrast.git"
