import os
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import PIL
import torch
from tqdm.auto import tqdm

from improved_diffusion.test_util import (mark_as_observed, tensor2avi,
                                          tensor2gif, tensor2mp4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        help=
        'A path to either a sample file or directory containing a long video split into multiple files.',
    )
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument(
        '--obs_length',
        type=int,
        default=0,
        help=
        'Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.',
    )
    parser.add_argument('--format',
                        type=str,
                        default='gif',
                        choices=['gif', 'mp4', 'avi'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--drop_file', type=int, default=0)
    parser.add_argument('--drop_frame', type=int, default=0)
    args = parser.parse_args()
    assert args.drop_file == 0 or args.drop_frame == 0
    args.path = Path(args.path)
    assert args.path.exists(), f'{args.path} does not exist.'
    out_path = (Path(args.out_dir) if args.out_dir is not None else
                args.path.parent) / f'long-{args.path.stem}.{args.format}'
    if not args.force:
        assert not out_path.exists(), f'{out_path} already exists.'

    if args.path.is_dir():
        filenames = list(args.path.glob('video_*.npy'))
        filenames.sort(key=lambda x: int(x.stem.split('_')[-1]))
    else:
        filenames = [args.path]
    if args.drop_file > 0:
        filenames = filenames[args.drop_file:]
    filenames = list(map(str, filenames))
    print('Filenames:', filenames)

    if len(filenames) == 0:
        print('Nothing to do.')
        exit(0)

    random_str = uuid.uuid4()

    video = np.concatenate([np.load(f) for f in filenames], axis=0)
    if args.obs_length > 0:
        mark_as_observed(video[:args.obs_length])
    # Add a final frame to mark the end of the video
    # final_frame = np.zeros_like(video[:1])
    # final_frame[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
    # video = np.concatenate([video, final_frame], axis=0)
    if args.drop_frame > 0:
        video = video[args.drop_frame:]
    if args.format == 'gif':
        tensor2gif(torch.tensor(video),
                   out_path,
                   drange=[0, 255],
                   random_str=random_str)
    elif args.format == 'mp4':
        tensor2mp4(torch.tensor(video),
                   out_path,
                   drange=[0, 255],
                   random_str=random_str)
    elif args.format == 'avi':
        tensor2avi(torch.tensor(video),
                   out_path,
                   drange=[0, 255],
                   random_str=random_str)
    else:
        raise ValueError(f'Unknown format {args.format}')
    print(f'Video (with {len(video)} frames) saved at {out_path}')
