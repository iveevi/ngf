FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
WORKDIR /home/
RUN apt update && apt install -y \
    libassimp-dev libglfw3-dev glslang-dev libvulkan-dev vulkan-validationlayers-dev \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    vim git wget ninja-build
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --upgrade matplotlib meshio pymeshlab seaborn \
    tqdm imageio largesteps polyscope gdown cmake \
    git+https://github.com/NVlabs/nvdiffrast
