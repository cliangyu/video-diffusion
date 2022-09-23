import argparse
import os

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('mask_path',
                    type=str,
                    help='Where to save the generated videos/coords.')
args = parser.parse_args()

batch_obs_indices, batch_lat_indices = map(np.array,
                                           torch.load(args.mask_path))
B, one, N = batch_lat_indices.shape
assert one == 1
batch_lat_indices = batch_lat_indices.reshape(B, N, 1)
assert batch_obs_indices.size == 0
batch_obs_indices = batch_obs_indices.reshape(B, N, 0)


def to_list(a):
    if type(a) in [int, np.int64, np.int32]:
        return a
    else:
        return [to_list(r) for r in a]


base_path, fname = os.path.split(args.mask_path)
new_path = os.path.join(base_path, 'independent-' + fname)
torch.save((to_list(batch_obs_indices), to_list(batch_lat_indices)), new_path)
