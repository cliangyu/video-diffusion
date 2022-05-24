import torch
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
from pathlib import Path
import uuid

from improved_diffusion.image_datasets import get_test_dataset
from improved_diffusion.test_util import mark_as_observed, tensor2gif, tensor2mp4, tensor2avi


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--do_n", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--obs_length", type=int, default=0, help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "avi"])
    args = parser.parse_args()

    out_dir = Path(args.samples_dir).parent / "video_arrays"
    out_dir.mkdir(exist_ok=True)
    random_str = uuid.uuid4()

    videos = []
    for video_i in range(args.do_n):
        data_idx = video_i
        gt_video, _ = dataset[data_idx]
        gt_video = gt_video.numpy()
        gt_video = (gt_video - gt_drange[0]) / (gt_drange[1] - gt_drange[0])  * 255
        gt_video = gt_video.astype(np.uint8)
        mark_as_observed(gt_video[:args.obs_length])
        videos.append([gt_video])
        for seed in range(args.n_seeds):
            filename = Path(args.samples_dir) / f"sample_000{video_i}-{seed}.npy"
            print(filename)
            video = np.load(filename)
            mark_as_observed(video[:args.obs_length])
            videos[-1].append(video)
        videos[-1] = np.concatenate(videos[-1], axis=-3)
    video = np.concatenate(videos, axis=-2)

    # Add a final frame to mark the end of the video
    final_frame = np.zeros_like(video[:1])
    final_frame[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
    video = np.concatenate([video, final_frame], axis=0)
    if args.format == "gif":
        tensor2gif(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    elif args.format == "mp4":
        tensor2mp4(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    elif args.format == "avi":
        tensor2avi(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    else:
        raise ValueError(f"Unknown format {args.format}")
    print(f"Saved to {out_path}")
