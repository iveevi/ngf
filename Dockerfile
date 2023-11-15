FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /home/
RUN apt update && apt install -y \
    libassimp-dev libglfw3-dev glslang-dev libvulkan-dev vulkan-validationlayers-dev \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    vim git wget ninja-build
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch torchvision
