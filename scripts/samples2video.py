import torch
import numpy as np
from argparse import ArgumentParser, Namespace
import PIL
from tqdm.auto import tqdm
import os
import imageio
from pathlib import Path
import uuid

from improved_diffusion.image_datasets import get_test_dataset


def mark_as_observed(images, color=[255, 0, 0]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


def tensor2pil(tensor, drange=[0,1]):
    """Given a tensor of shape (Bx)3xwxh with pixel values in drange, returns a PIL image
       of the tensor. Returns a list of images if the input tensor is a batch.
    Args:
        tensor: A tensor of shape (Bx)3xwxh
        drange (list, optional): Range of pixel values in the input tensor. Defaults to [0,1].
    """
    assert tensor.ndim == 3 or tensor.ndim == 4
    if tensor.ndim == 3:
        return tensor2pil(tensor.unsqueeze(0), drange=drange)[0]
    img_batch = tensor.cpu().numpy().transpose([0, 2, 3, 1])
    img_batch = (img_batch - drange[0]) / (drange[1] - drange[0])  * 255 # img_batch with pixel values in [0, 255]
    img_batch = img_batch.astype(np.uint8)
    return [PIL.Image.fromarray(img) for img in img_batch]

def tensor2gif(tensor, path, drange=[0, 1], random_str=""):
    frames = tensor2pil(tensor, drange=drange)
    tmp_path = f"/tmp/tmp_{random_str}.png"
    res = []
    for frame in frames:
        frame.save(tmp_path)
        res.append(imageio.imread(tmp_path))
    imageio.mimsave(path, res)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--add_gt", action="store_true")
    parser.add_argument("--do_n", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--obs_length", type=int, default=0,
                        help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    args = parser.parse_args()

    drange = [0, 255]

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
    filenames = list(filenames) + list(Path(args.samples_dir).glob("video_*.npy"))
    #print([str(f).replace('-', '.').split('.') for f in filenames])
    #filenames = [f for f in filenames if int(str(f).replace('-', '.').split('.')[-2]) < args.n_seeds]
    print(filenames)
    for filename in sorted(filenames)[:args.do_n]:
        video_name = filename.stem
        data_idx = int(video_name.split("_")[1].split("-")[0])
        gif_path = out_dir / f"{video_name}.gif"
        if gif_path.exists():
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
        tensor2gif(torch.tensor(video), gif_path, drange=drange, random_str=random_str)
