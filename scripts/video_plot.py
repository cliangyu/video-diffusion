import glob
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_frames_from_gif(path, indices):
    gif = Image.open(path)
    frames = []
    for index in indices:
        gif.seek(index)
        frames.append(np.asarray(gif))
    return frames


name = 'mazes-autoreg'
# name = 'mazes-hierarchy-2'
# name = 'minerl-autoreg'
# name = 'minerl-hierarchy-2'
#    f"/ubc/cs/research/plai-scratch/wsgh/syncing-plots/basic-videos/{name}-{i}.gif"
#    for i in [0, 1, 2, 5, 6, 7]
# ]

parser = ArgumentParser()
parser.add_argument('--gif_dir', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=2)
parser.add_argument('--n_videos', type=int, default=3)
parser.add_argument('--T', type=int, default=500)
args = parser.parse_args()

T = args.T  # 1000 if 'carla' in name else 300 if 'mazes' in name else 500
# video_paths = [
video_paths = glob.glob(os.path.join(args.gif_dir, '*.gif'))
print(video_paths)
to_keep = []
n_per_vid = {}
for path in video_paths:
    splitted = path.replace('_', '-').replace('.', '-').split('-')
    video_id = int(splitted[-3])
    seed = int(splitted[-2])
    if video_id < args.n_videos:
        if video_id not in n_per_vid:
            n_per_vid[video_id] = 0
        if n_per_vid[video_id] < args.n_samples:
            to_keep.append(path)
            n_per_vid[video_id] += 1
video_paths = sorted(to_keep)
print(video_paths)

indices_to_plot = [int(i) for i in np.linspace(1, T - 1, 14)]
print(
    dict(nrows=len(video_paths),
         ncols=len(indices_to_plot),
         figsize=(7.5, 3.7)))
fig, axes = plt.subplots(nrows=len(video_paths),
                         ncols=len(indices_to_plot),
                         figsize=(7.5, 3.7))
plt.subplots_adjust(wspace=0, hspace=0.1)
for ax_row, path in zip(axes, video_paths):
    frames = load_frames_from_gif(path, indices_to_plot)
    for ax, frame, index in zip(ax_row, frames, indices_to_plot):
        # if index < 36:
        #    c = np.array([10, 210, 255])
        #    frame[:2, :] = c
        #    frame[-2:, :] = c
        #    frame[:, :2] = c
        #    frame[:, -2:] = c
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])

for ax, t in zip(axes[-1], indices_to_plot):
    ax.set_xlabel(t + 1, fontsize=8)

fig.savefig(os.path.join(args.gif_dir, 'array.pdf'), bbox_inches='tight')
