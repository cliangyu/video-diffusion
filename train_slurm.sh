#!/bin/bash

GPU=4
CPU=32
node=73
PORT=29500
jobname=flexdiff

export NO_MPI=1

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python -m torch.distributed.run \
  --nproc_per_node ${GPU} \
  scripts/video_train.py \
  --dataset carla_no_traffic \
  --batch_size 1 \
  --max_frames 20 \
  --sample_interval 100