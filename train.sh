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
  --batch_size 2 \
  --max_frames 20 \
  --observed_frames x_0 \
  --num_res_blocks 1 \
  --num_workers 16 \
  --data_path /workspace/data

  scripts/video_train.py \
  --dataset carla_no_traffic \
  --batch_size 2 \
  --max_frames 20 \
  --observed_frames x_t_minus_1 \
  --num_res_blocks 1 \
  --num_workers 16 \
  --data_path /workspace/data

  scripts/video_train.py \
  --dataset carla_no_traffic \
  --batch_size 2 \
  --max_frames 20 \
  --observed_frames x_random \
  --num_res_blocks 1 \
  --num_workers 16 \
  --data_path /workspace/data

    scripts/video_train.py \
  --dataset carla_no_traffic \
  --batch_size 2 \
  --max_frames 20 \
  --observed_frames x_t \
  --num_res_blocks 1 \
  --num_workers 16 \
  --data_path /workspace/data
