# The code is mostly based on https://gist.github.com/vaibhavsaxena11/97b2d0a195c08ab2ed75cebb7d763799

# 3-class accuracy for GQN Mazes
# class 1: room_stay (agent stays in the room and never enters the hallway)
# class 2: hallway_enter_stay (agent goes from the room to the hallway but does not exit back into the room)
# class 3: hallway_enter_recover (agent goes from room to hallway and then back to the room) --> difficult to predict for the model as it has to remember the orig room patterns for a longer duration
###
import json
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import OrderedDict

import cv2
import lpips as lpips_metric
import numpy as np
import torch
# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from tqdm.auto import tqdm

import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion import test_util
from improved_diffusion.image_datasets import get_test_dataset


class LazyDataFetch:
    def __init__(
        self,
        dataset,
        eval_dir,
        obs_length,
        dataset_drange,
        drop_obs=True,
        num_samples=None,
    ):
        """A class to handle loading sampled videos and their corresponding gt
        videos from the dataset.

        Arguments:
            drop_obs: if True, drops the observed part of the videos from the output pairs of videos.
            num_samples: if not None, asserts that all the videos have at least num_samples generated samples.
        """
        self.obs_length = obs_length
        self.drop_obs = drop_obs
        samples_dir = Path(eval_dir) / 'samples'
        assert samples_dir.exists(
        ), f'Samples dir {samples_dir} does not exist.'
        filenames = [(x,
                      [int(num) for num in x.stem.split('_')[-1].split('-')])
                     for x in samples_dir.glob('sample_*.npy')]
        # filenames has the following structure: [(filename, (video_idx, sample_idx))]
        filenames.sort(key=lambda x: x[1][0])
        # Arrange all the filenames in a dictionary from the test set index to a list of filenames (one filename for each generated sample).
        self.filenames_dict = defaultdict(list)
        for item in filenames:
            filename, (video_idx, sample_idx) = item
            self.filenames_dict[video_idx].append(filename)
        if num_samples is not None:
            for idx, filenames in self.filenames_dict.items():
                assert (
                    len(filenames) >= num_samples
                ), f'Expected at least {num_samples} samples for each video, but found {len(filenames)} for video #{idx}'
        self.keys = list(self.filenames_dict.keys())
        self.dataset = dataset
        self.dataset_drange = dataset_drange
        assert self.dataset_drange[1] > self.dataset_drange[0]

    def __getitem__(self, idx):
        # Returns a tuple of (gt video, [list of sampled videos])
        # Each video has shape of TxCx3xHxW
        video_idx = self.keys[idx]
        filename_list = self.filenames_dict[video_idx]
        preds = {
            str(filename): (np.load(filename) / 255.0).astype(np.float32)
            for filename in filename_list
        }  # pred with pixel values in [0, 1]
        gt = self.dataset[video_idx][0].numpy()
        gt = (gt - self.dataset_drange[0]) / (
            self.dataset_drange[1] - self.dataset_drange[0]
        )  # gt with pixel values in [0, 1]
        gt = gt.astype(np.float32)
        if self.drop_obs:
            gt = gt[self.obs_length:]
            preds = {k: x[self.obs_length:] for k, x in preds.items()}
        return {'gt': gt, 'preds': preds}

    def __len__(self):
        return len(self.keys)

    def get_num_samples(self):
        # Returns the number of samples per test video in the database. Assumes all test videos have the same number of samples
        return len(self[0]['preds'])

    @property
    def T(self):
        res = list(self[0]['preds'].values())[0].shape[0]
        if self.drop_obs:
            res += self.obs_length
        return res


def _smooth_seq(seqs):
    # seqs shape: (..., N)
    kernel = [i / 5.0
              for i in range(1, 6)] + [i / 5.0 for i in reversed(range(1, 5))]
    ss = np.zeros(
        list(seqs.shape[:-1]) +
        [seqs.shape[-1] + len(kernel) - (len(kernel) % 2)])
    ss[..., len(kernel) // 2:-(len(kernel) // 2)] = seqs
    out = np.zeros_like(seqs)
    for i in range(out.shape[-1]):
        if i in range((len(kernel) // 2)):
            k = kernel[(len(kernel) // 2) - i:]
        elif i in range(out.shape[-1] - (len(kernel) // 2), out.shape[-1]):
            k = kernel[:-(i + (len(kernel) // 2) - out.shape[-1] + 1)]
        else:
            k = kernel
        out[..., i] = np.dot(ss[..., i:i + len(kernel)], kernel) / np.sum(k)
    return out


def _count_hallway_pixels(seqs):
    num_green_pixels = []
    for seq in seqs:
        green_pixels = []
        for image in seq:
            strip = image[14:45]
            hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (50, 25, 25), (70, 255, 255))  # green
            mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
            green_pixels.append(len(np.nonzero(mask)[0]))
        num_green_pixels.append(green_pixels)
    return _smooth_seq(np.array(num_green_pixels))


def verify_hallway(seqs, entry_thresh, out_thresh):
    """
    seqs shape: (B, T, 64, 64, 3)
    Returns:
        hallway : shape (B,T)
            bool indicating when a scene is inside a hallway or not
        room_stay : shape (B,)
            0/1 per seq; indicator 1 if agent stays inside the room throughout the seq
        hallway_enter_stay : shape (B,)
            0/1 per seq; indicator 1 if agent goes from room-->hallway and stays in the hallway
        hallway_enter_recover : shape (B,)
            int per seq; number of times the scene went from room-->hallway-->room
    """
    seqs = np.array(seqs)
    hallway_pixels = _count_hallway_pixels(seqs)
    # Bools indicating in/out of hallway -->
    hallway = np.zeros_like(hallway_pixels)
    # Indicator per seq for the three classes of sequences -->
    room_stay = np.zeros(seqs.shape[0])  # class 1
    hallway_enter_stay = np.zeros(seqs.shape[0])  # class 2
    hallway_enter_recover = np.zeros(seqs.shape[0])  # class 3
    for b in range(seqs.shape[0]):
        in_hallway = False
        room_stay[b] = 1.0
        hallway_enter_stay[b] = 0.0
        hallway_stay_probe_on = True
        recovery_probe_on = False
        for t in range(seqs.shape[1]):
            if in_hallway:
                if hallway_pixels[b, t] > out_thresh:
                    hallway[b, t] = 1.0
                else:
                    in_hallway = False
                    hallway_enter_stay[b] = 0.0
                    hallway_stay_probe_on = False
                    if recovery_probe_on:
                        hallway_enter_recover[b] += 1
                        recovery_probe_on = False
            else:
                if hallway_pixels[b, t] > entry_thresh:
                    hallway[b, t] = 1.0
                    in_hallway = True
                    room_stay[b] = 0.0
                    if hallway_stay_probe_on:
                        hallway_enter_stay[b] = 1.0
                    recovery_probe_on = True  # and t != 0
    return hallway, room_stay, hallway_enter_stay, hallway_enter_recover


def print_metrics(name, metrics):
    count = 0
    for cname, (met, pos_idxs) in metrics.items():
        pos = np.squeeze(np.array([met[idx] for idx in pos_idxs]))
        count += np.sum((pos > 0).astype(int))
    print('{} : acc={}/{} = {}%'.format(name, count, len(met),
                                        count / len(met) * 100))


def get_single_stats(metrics):
    count = 0
    for cname, (met, pos_idxs) in metrics.items():
        pos = np.squeeze(np.array([met[idx] for idx in pos_idxs]))
        count += np.sum((pos > 0).astype(int))
    return count / len(met)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument(
        '--obs_length',
        type=int,
        default=36,
        help='Number of observed frames. Default is 36.',
    )
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    args.dataset = 'mazes_cwvae'
    args.T = 300
    args.batch_size = 16
    args.num_samples = 5
    args.dataset_partition = 'test'
    # Load dataset
    dataset = locals()[f'get_{args.dataset_partition}_dataset'](
        dataset_name=args.dataset
    )  # Load the full-length videos. We'll use the first T frames for evaluation, however.
    drange = [-1, 1]  # Range of dataset's pixel values
    data_fetch = LazyDataFetch(
        dataset=dataset,
        eval_dir=args.eval_dir,
        obs_length=args.obs_length,
        dataset_drange=drange,
        num_samples=args.num_samples,
    )
    assert args.T <= data_fetch.T

    def get_sample(preds_dict, idx):
        keys = list(preds_dict.keys())
        keys = {int(Path(x).stem.split('-')[-1]): x for x in keys}
        return preds_dict[keys[idx]]

    seqs_gt = np.stack([
        data_fetch[i]['gt'].transpose([0, 2, 3, 1])
        for i in range(len(data_fetch))
    ])
    seqs_gt = (seqs_gt * 255).astype(np.uint8)

    hallway_entry_thresh = 1000
    hallway_out_thresh = 500

    hallways_gt, room_stay_gt, hallway_stay_gt, recovery_gt = verify_hallway(
        seqs_gt, hallway_entry_thresh, hallway_out_thresh)
    room_stay_idxs = np.nonzero((room_stay_gt > 0).astype(int))[0]
    hallway_stay_idxs = np.nonzero((hallway_stay_gt > 0).astype(int))[0]
    recovery_idxs = np.nonzero((recovery_gt > 0).astype(int))[0]

    print('Num examples Class 1:', len(room_stay_idxs))
    print('Num examples Class 2:', len(hallway_stay_idxs))
    print('Num examples Class 3:', len(recovery_idxs))

    print('3-class accuracies:')
    print_metrics(
        'GT',
        {
            'room stay': (room_stay_gt, room_stay_idxs),
            'hallway enter stay': (hallway_stay_gt, hallway_stay_idxs),
            'hallway enter recover': (recovery_gt, recovery_idxs),
        },
    )

    acc_list = []
    for idx in range(args.num_samples):
        seqs_pred = np.stack([
            get_sample(data_fetch[i]['preds'], idx=idx).transpose([0, 2, 3, 1])
            for i in range(len(data_fetch))
        ])
        seqs_pred = (seqs_pred * 255).astype(np.uint8)
        (
            hallways_pred,
            room_stay_pred,
            hallway_stay_pred,
            recovery_pred,
        ) = verify_hallway(seqs_pred, hallway_entry_thresh, hallway_out_thresh)
        acc_list.append(
            get_single_stats({
                'room stay': (room_stay_pred, room_stay_idxs),
                'hallway enter stay': (hallway_stay_pred, hallway_stay_idxs),
                'hallway enter recover': (recovery_pred, recovery_idxs),
            }))
    print(
        f'{np.mean(acc_list) * 100}% +- {np.std(acc_list) / np.sqrt(args.num_samples - 1) * 100}'
    )
    print(f'{np.max(acc_list) * 100}%')
