from nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

run apt-get update && apt-get install -y \
    python3\
    python3-pip \
    git\
    && rm -rf /var/lib/apt/lists/*
run pip3 install --upgrade pip
run pip3 install --upgrade cython
run pip3 install \
    torch==1.2\
    torchvision==0.4.0\
    scipy\
    pillow==6.2.1\
    sklearn\
    tqdm\
    torchsummary\
    matplotlib\
    opencv-python-headless\
    pandas\
    scikit-image\
    'git+https://github.com/facebookresearch/fvcore'\
    av\
    psutil\
    tensorboardX

# install facebook Detectron2 API
#run pip3 install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
#run git clone https://github.com/facebookresearch/detectron2 detectron2_repo
#run pip3 install -e detectron2_repo

workdir /home/zhengwei
cmd ["/bin/bash"]
