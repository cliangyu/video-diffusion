import torch
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
import os
from pathlib import Path
import uuid

from improved_diffusion.image_datasets import get_test_dataset, mark_as_observed, tensor2gif, tensor2mp4, tensor2avi


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
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

    out_dir = Path(args.samples_dir).parent / out_dir
    out_dir.mkdir(exist_ok=True)
    random_str = uuid.uuid4()

    filenames = Path(args.samples_dir).glob("sample_*.npy")
    #print([str(f).replace('-', '.').split('.') for f in filenames])
    filenames = [f for f in filenames if int(str(f).replace('-', '.').split('.')[-2]) < args.n_seeds]
    print(filenames)
    for filename in sorted(filenames)[:args.do_n]:
        video_name = filename.stem
        data_idx = int(video_name.split("_")[1].split("-")[0])
        out_path = out_dir / f"{video_name}.{args.format}"
        if out_path.exists():
            print(f"Skipping {video_name}. Already exists.")
            continue
        print(f"Processing {video_name}")
        video = np.load(filename)
        if args.obs_length > 0:
            mark_as_observed(video[:args.obs_length])
        if args.add_gt:
            gt_drange = [-1, 1]
            gt_video, _ = dataset[data_idx]
            gt_video = gt_video.numpy()
            gt_video = (gt_video - gt_drange[0]) / (gt_drange[1] - gt_drange[0])  * 255
            gt_video = gt_video.astype(np.uint8)
            video = np.concatenate([video, gt_video], axis=-2)
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
