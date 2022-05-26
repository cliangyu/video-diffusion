import numpy as np
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def load_frames_from_gif(path, indices):
    gif = Image.open(path)
    frames = []
    for index in indices:
        gif.seek(index)
        frames.append(np.asarray(gif))
    return frames

name = 'mazes-autoreg'
#name = 'mazes-hierarchy-2'
#name = 'minerl-autoreg'
#name = 'minerl-hierarchy-2'
T = 300 if 'mazes' in name else 500
video_paths = [
    f"/ubc/cs/research/plai-scratch/wsgh/syncing-plots/basic-videos/{name}-{i}.gif"
    for i in [0, 1, 2, 5, 6, 7]
]

indices_to_plot = [int(i) for i in np.linspace(1, T-1, 14)]
fig, axes = plt.subplots(nrows=len(video_paths), ncols=len(indices_to_plot), figsize=(7.5, 3.7))
plt.subplots_adjust(wspace=0, hspace=0.1)
for ax_row, path in zip(axes, video_paths):
    frames = load_frames_from_gif(path, indices_to_plot)
    for ax, frame, index in zip(ax_row, frames, indices_to_plot):
        if index < 36:
            c = np.array([10, 210, 255])
            frame[:2, :] = c
            frame[-2:, :] = c
            frame[:, :2] = c
            frame[:, -2:] = c
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])

for ax, t in zip(axes[-1], indices_to_plot):
    ax.set_xlabel(t+1, fontsize=8)

fig.savefig(f'visualisations/{name}.pdf', bbox_inches='tight')
