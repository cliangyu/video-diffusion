import torch
import numpy as np
from argparse import ArgumentParser, Namespace
import PIL
from tqdm.auto import tqdm
import os
import imageio
from pathlib import Path
import uuid
from PIL import Image


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
    parser.add_argument("path", type=str, help="A path to either a sample file or directory containing a long video split into multiple files.")
    parser.add_argument("--obs_length", type=int, default=0,
                        help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--drop_file", type=int, default=0)
    parser.add_argument("--drop_frame", type=int, default=0)
    args = parser.parse_args()
    assert args.drop_file==0 or args.drop_frame==0
    args.path = Path(args.path)
    assert args.path.exists(), f"{args.path} does not exist."
    out_path = args.path.parent / f"{args.path.stem}.gif"
    if not args.force:
        assert not out_path.exists(), f"{out_path} already exists."

    if args.path.is_dir():
        filenames = list(args.path.glob("video_*.npy"))
        filenames.sort(key=lambda x: int(x.stem.split("_")[-1]))
    else:
        filenames = [args.path]
    if args.drop_file > 0:
        filenames = filenames[args.drop_file:]
    print("Filenames:", filenames)

    if len(filenames) == 0:
        print("Nothing to do.")
        exit(0)

    random_str = uuid.uuid4()

    video = np.concatenate([np.load(f) for f in filenames], axis=0)
    if args.obs_length > 0:
        mark_as_observed(video[:args.obs_length])
    # Add a final frame to mark the end of the video
    final_frame = np.zeros_like(video[:1])
    final_frame[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
    video = np.concatenate([video, final_frame], axis=0)
    if args.drop_frame > 0:
        video = video[args.drop_frame:]
    tensor2gif(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    print(f"Video (with {len(video)} frames) saved at {out_path}")