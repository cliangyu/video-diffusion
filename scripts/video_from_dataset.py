import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from improved_diffusion.image_datasets import (get_test_dataset,
                                               get_train_dataset,
                                               get_variable_length_dataset)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('-i', type=int, required=True)
    parser.add_argument('-o', '--out', default='videos/datasets')
    parser.add_argument(
        '--dataset_partition',
        default='test',
        choices=['train', 'test', 'variable_length'],
    )
    parser.add_argument(
        '--length',
        type=int,
        default=None,
        help=
        'Length of the videos. If not specified, it will save the whole video.',
    )
    args = parser.parse_args()
    args.out = (Path(args.out) / args.dataset /
                f'{args.dataset_partition}_{args.i}_{args.length}.npy')
    assert not args.out.exists(), f'{args.out} already exists.'

    drange = [-1, 1]  # Range of the generated samples' pixel values

    dataset = locals()[f'get_{args.dataset_partition}_dataset'](
        dataset_name=args.dataset)
    print(f'Dataset size = {len(dataset)}, taking video #{args.i}')

    # Save the video as a file
    video, _ = dataset[args.i]
    video = video.numpy()
    if args.length is None:
        args.length = len(video)
    video = video[:args.length]
    video = ((video - drange[0]) / (drange[1] - drange[0]) * 255
             )  # video with pixel values in [0, 255]
    video = video.astype(np.uint8)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    np.save(str(args.out), video)
    print(f'Saved video to {args.out}')
