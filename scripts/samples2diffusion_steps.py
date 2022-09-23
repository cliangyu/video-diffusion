import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
from pathlib import Path
import uuid

from improved_diffusion.image_datasets import get_test_dataset
from improved_diffusion.test_util import mark_as_observed, tensor2gif, tensor2mp4, tensor2avi


def mark_as_not_observed(images, color=[0, 0, 0]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c
        
        
def image_grid(array, ncols=4):
    index, height, width, channels = array.shape
    nrows = index//ncols
    
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
            .swapaxes(1,2)
            .reshape(height*nrows, width*ncols, channels))
    
    return img_grid


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--add_gt", action="store_true")
    parser.add_argument("--do_n", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--obs_length", type=int, default=0,
                        help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "avi"])
    args = parser.parse_args()

    if args.add_gt:
        assert args.dataset is not None
        dataset = get_test_dataset(args.dataset)
        out_dir = "videos_and_gt"
    else:
        out_dir = "videos"

    out_dir = (Path(args.out_dir) if args.out_dir is not None else Path(args.samples_dir).parent) / out_dir
    out_dir.mkdir(exist_ok=True)
    random_str = uuid.uuid4()

    filenames = Path(args.samples_dir).glob("all_timestep_sample_*.npy")
    filenames = list(filenames) + list(Path(args.samples_dir).glob("video_*.npy"))
    print(filenames)
    for filename in sorted(filenames)[:args.do_n]:
        video_name = filename.stem
        data_idx = int(video_name.split("_")[-1].split("-")[0])
        out_path = out_dir / f"{video_name}.{args.format}"
        if out_path.exists():
            print(f"Skipping {video_name}. Already exists.")
            continue
        print(f"Processing {video_name}")
        try:
            video = np.load(filename)
        except PermissionError:
            print('Permission denied.')
            continue
        if args.obs_length > 0:
            mark_as_observed(video[:,:args.obs_length,:,:,:])
            mark_as_not_observed(video[:,args.obs_length:,:,:,:])
            
        visualize_step = 20
        timesteps = list(range(video.shape[0]))[::visualize_step]
        timesteps.append(video.shape[0]-1)    
        vis_video = video[timesteps,:,:,:,:]
        D, T, C, H, W = vis_video.shape
        vis_video = vis_video.reshape([-1, C, H, W])
        vis_video = np.einsum('NCHW->NHWC', vis_video)

        result = image_grid(vis_video, ncols=T)
        fig = plt.figure(figsize=(20., 10.))
        plt.imshow(result)
        plt.margins(x=0, y=0)
        sampling_method, max_frames, step_size, total_frames, observed_frames = args.samples_dir.split('/')[-2].split('_')[:5]
        
        plt.title(f"Sampling method: {sampling_method} Sliding window: {max_frames} # Generated frames: {step_size} Video length: {total_frames} Ground truth: {observed_frames}")
        out_path = out_dir / f"{video_name}.png"
        plt.savefig(out_path, dpi=300)
        print(f"Saved to {out_path}")
        
