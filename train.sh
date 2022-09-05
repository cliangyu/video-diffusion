#!/bin/bash

GPU=4
CPU=32
node=73
PORT=29500
jobname=flexdiff

export NO_MPI=1

python -m torch.distributed.run \
  --nproc_per_node ${GPU} \
  scripts/video_train.py \
  --dataset carla_no_traffic \
  --batch_size 1 \
  --max_frames 20 \
  --sample_interval 10000 \
  --save_interval 10000 \
  --observed_frames x_0 \
  --data_path /media/ntu/volume1/home/s121md302_06/data/no-traffic
