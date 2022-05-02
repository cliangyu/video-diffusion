import numpy as np
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from improved_diffusion.train_util import concat_images_with_padding


def plot_video(args):
    video = np.load(args.path)
    # pad_dim_h = pad_dim_ = 2
    # pad_val = 0
    # img = concat_images_with_padding(
    #     [concat_images_with_padding(video, horizontal=True, pad_dim=pad_dim_h, pad_val=pad_val, pad_ends=pad_ends) for vid in videos],
    #     horizontal=False, pad_dim=pad_dim_v, pad_val=pad_val, pad_ends=pad_ends,
    # )
    #img = concat_images_with_padding(video, horizontal=True, pad_dim)
    #Image.fromarray(img).save(args.save_as)
    video = video.transpose(0, 2, 3, 1)
    frame_indices = np.concatenate([np.arange(0, 35, 5), np.arange(36, 300, 20)])
    video = video[frame_indices]
    fig, axes = plt.subplots(ncols=len(video))
    for ax, frame, i in zip(axes, video, frame_indices):
        ax.imshow(frame)
        ax.set_title(str(i), fontsize=8)
        ax.axis('off')
    plt.savefig(args.save_as, bbox_inches='tight')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--save_as", type=str, required=True)
    args = parser.parse_args()

    plot_video(args)
