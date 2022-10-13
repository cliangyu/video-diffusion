FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install screen -y
RUN apt-get install psmisc -y
RUN apt-get install python3.8 python3-pip -y
RUN apt-get install python-is-python3 -y
RUN apt-get install unrar

RUN pip install --upgrade pip
RUN pip install jupyter
RUN pip install wandb

RUN pip install absl-py>=0.12.0
RUN pip install numpy>=1.21.5
RUN pip install torch==1.10.1 \
torchvision \
opencv-python \
dopamine-rl \
ml-collections==0.1.0 \
tqdm \
h5py \
tensorflow_datasets \
blobfile \
imageio \
moviepy

WORKDIR /workspace/dedt
ENV PYTHONPATH /workspace/dedt
ENV WANDB_API_KEY c332c7a2cc204e6848aac6058bcfcee460a38f0e
RUN apt-get install wget -y
RUN cd ~ && mkdir Roms && cd Roms && wget http://www.atarimania.com/roms/Roms.rar
RUN echo -e "a\na\na\n" | unrar e ~/Roms/Roms.rar ~/Roms
RUN python -m atari_py.import_roms ~/Roms
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install jupyterlab
