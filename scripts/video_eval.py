from collections import defaultdict
from typing import OrderedDict
import torch
import numpy as np
from argparse import ArgumentParser
import os
from tqdm.auto import tqdm
from pathlib import Path
import json
import pickle
# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips as lpips_metric
from collections import defaultdict
try:
   import tensorflow as tf
except ModuleNotFoundError:
    print("WARNING: failed tensorflow import")

from improved_diffusion.image_datasets import get_test_dataset
try:
    import improved_diffusion.frechet_video_distance as fvd
except ModuleNotFoundError:
    print("WARNING: failed fvd import")


def compute_FVD(vid1, vid2):
    # Batch has to be divisible by 16
    assert vid1.shape[0] % 16 == 0 
    assert vid2.shape[0] % 16 == 0
    
    # assert channel is last dimension
    # and trust that shape is B, T, H, W, C
    assert vid1.shape[4] == 3
    assert vid2.shape[4] == 3
    
    with tf.compat.v1.Graph().as_default():
        vid1 = tf.compat.v1.convert_to_tensor(vid1, np.uint8)
        vid2 = tf.compat.v1.convert_to_tensor(vid2, np.uint8)
        
        result = fvd.calculate_fvd(
            fvd.create_id3_embedding(fvd.preprocess(vid1,(224, 224))),
            fvd.create_id3_embedding(fvd.preprocess(vid2,(224, 224)))
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            return sess.run(result)


class LazyDataFetch:
    def __init__(self, dataset, samples_dir, obs_length, dataset_drange):
        self.obs_length = obs_length
        samples_dir = Path(samples_dir)
        assert samples_dir.exists()
        filenames = [(x, [int(num) for num in x.stem.split("_")[-1].split("-")]) for x in samples_dir.glob("sample_*.npy")]
        # filenames has the following structure: [(filename, (video_idx, sample_idx))]
        filenames.sort(key=lambda x: x[1][0])
        self.filenames_dict = defaultdict(list)
        for item in filenames:
            filename, (video_idx, sample_idx) = item
            self.filenames_dict[video_idx].append(filename)
        self.keys = list(self.filenames_dict.keys())
        self.dataset = get_test_dataset(dataset)
        self.dataset_drange = dataset_drange
        assert self.dataset_drange[1] > self.dataset_drange[0]

    def __getitem__(self, idx):
        # Returns a tuple of (gt video, [list of sampled videos])
        # Each video has shape of TxCx3xHxW
        video_idx = self.keys[idx]
        filename_list = self.filenames_dict[video_idx]
        preds = [(np.load(filename) / 255.0).astype(np.float32) for filename in filename_list] # pred with pixel values in [0, 1]
        gt = dataset[video_idx][0].numpy()
        gt = (gt - self.dataset_drange[0]) / (self.dataset_drange[1] - self.dataset_drange[0]) # gt with pixel values in [0, 1]
        gt = gt.astype(np.float32)
        return {"gt":gt[self.obs_length:],
                "preds": [x[self.obs_length:] for x in preds]}

    def __len__(self):
        return len(self.keys)

    def get_num_samples(self):
        # Returns the number of samples per test video in the database. Assumes all test videos have the same number of samples
        return len(self[0]["preds"])


def compute_metrics(preds, gts):
    """ Computes SSIM and PSNR metrics between the given prediction
    and ground truth videos.

    Args:
        preds (np.array): Shape: BxTxCxHxW
        gts (np.array): Shape: BxTxCxHxW
    """
    B, T, C, H, W = gts.shape
    ssim = np.zeros((B, T))
    psnr = np.zeros((B, T))
    for i in tqdm(range(B)):
        for t in range(T):
            for c in range(C):
                ssim[i, t] += ssim_metric(gts[i, t, c], preds[i, t, c])
                psnr[i, t] += psnr_metric(gts[i, t, c], preds[i, t, c])
            psnr[i, t] /= C
            ssim[i, t] /= C
    return {"ssim": ssim,
            "psnr": psnr}


def compute_metrics_lazy(data_fetch, T, num_samples, C=3):
    num_videos = len(data_fetch)
    ssim = np.zeros((num_videos, num_samples, T))
    psnr = np.zeros((num_videos, num_samples, T))
    for i in tqdm(range(num_videos)):
        data = data_fetch[i]
        gt = data["gt"]
        preds = data["preds"]
        assert len(preds) >= num_samples, f"Expected at least {num_samples} video prediction samples. Found {len(preds)} for video #{i}"
        preds = preds[:num_samples]
        for k, pred in enumerate(preds):
            for t in range(T):
                for c in range(C):
                    ssim[i, k, t] += ssim_metric(gt[t, c], pred[t, c])
                    psnr[i, k, t] += psnr_metric(gt[t, c], pred[t, c])
                psnr[i, k, t] /= C
                ssim[i, k, t] /= C
    return {"ssim": ssim,
            "psnr": psnr}


@torch.no_grad()
def compute_lpips_lazy(data_fetch, T, num_samples, device="cuda"):
    num_videos = len(data_fetch)
    lpips = np.zeros((num_videos, num_samples, T))
    loss_fn = lpips_metric.LPIPS(net='alex', spatial=False).to(device)
    for i in tqdm(range(num_videos)):
        data = data_fetch[i]
        gt = data["gt"]
        preds = data["preds"]
        assert len(preds) >= num_samples, f"Expected at least {num_samples} video prediction samples. Found {len(preds)} for video #{i}"
        preds = preds[:num_samples]
        gt = gt[:T]
        preds = np.stack([pred[:T] for pred in preds])
        # Change pixel normalization from [0, 1] to [-1, 1]
        gt = gt * 2 - 1
        preds = preds * 2 - 1
        # Convert to tensors on the correct device
        gt = torch.tensor(gt).to(device)
        for k, pred in enumerate(preds):
            pred = torch.tensor(pred).to(device)
            lpips[i, k, :] = loss_fn(gt, pred).flatten().cpu().numpy()
    return {"lpips": lpips}



def compute_fvd_lazy(data_fetch, T, num_samples):
    num_videos = len(data_fetch)
    fvd = np.zeros((num_videos // 16, num_samples))
    for i in tqdm(range(0, num_videos, 16)):
        data = [data_fetch[j] for j in range(i, i+16)]
        gt_batch = [x["gt"] for x in data]
        preds_batch = [np.stack(x["preds"][:num_samples]) for x in data]
        gt_batch = np.stack(gt_batch)[:, :T] # BxTxCxHxW
        preds_batch = np.stack(preds_batch)[:, :, :T] # BxNxTxCxHxW
        assert preds_batch.shape[1] == num_samples, f"Expected at least {num_samples} video prediction samples."
        # Convert image pixels to bytes
        gt_batch = (gt_batch * 255).astype("uint8")
        preds_batch = (preds_batch * 255).astype("uint8")

        gt_batch = np.moveaxis(gt_batch, 2, 4) # B, T, H, W, C
        preds_batch = np.moveaxis(preds_batch, 3, 5) # B, T, H, W, C
        # Convert to tensors on the correct device
        for k in range(num_samples):
            pred_batch = preds_batch[:, k]
            print(pred_batch.shape, gt_batch.shape)
            x = compute_FVD(pred_batch, gt_batch)
            print(x)
            fvd[i // 16, k] = x
    return {"fvd": fvd}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--modes", nargs="+", type=str, default=["all"],
                        choices=["ssim", "psnr", "lpips", "fvd", "all"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--obs_length", type=int, required=True,
                        help="Number of observed frames.")
    parser.add_argument("--T", type=int, default=None,
                        help="Video length. If not given, the length of the dataset is used.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of generated samples per test video.")
    args = parser.parse_args()

    if "all" in args.modes:
        args.modes = ["ssim", "psnr", "lpips"]
    # Load dataset
    dataset = get_test_dataset(args.dataset)
    drange = [-1, 1] # Range of dataset's pixel values
    data_fetch = LazyDataFetch(dataset=args.dataset,
                               samples_dir=args.samples_dir,
                               obs_length=args.obs_length,
                               dataset_drange=drange)
    if args.num_samples is None:
        args.num_samples = data_fetch.get_num_samples()
    if args.T is None:
        args.T = data_fetch[0]["preds"][0].shape[0]
    else:
        assert args.T <= data_fetch[0]["preds"][0].shape[0]

    # Check if metrics have already been computed
    name = f"new_metrics_{len(data_fetch)}-{args.num_samples}-{args.T}"
    pickle_path = Path(args.samples_dir) / f"{name}.pkl"
    if pickle_path.exists():
        metrics_pkl = pickle.load(open(pickle_path, "rb"))
        args.modes = [mode for mode in args.modes if mode not in metrics_pkl]
    else:
        metrics_pkl = {}
    print(f"Modes: {args.modes}")
    if len(args.modes) == 0:
        print("No metrics to compute.")
        exit(0)

    # Compute metrics
    new_metrics = {}
    if "ssim" in args.modes or "psnr" in args.modes:
        new_metrics.update(compute_metrics_lazy(
            data_fetch=data_fetch,
            T=args.T,
            num_samples=args.num_samples))
    if "lpips" in args.modes:
        new_metrics.update(compute_lpips_lazy(
            data_fetch=data_fetch,
            T=args.T,
            num_samples=args.num_samples))
    if "fvd" in args.modes:
        new_metrics.update(compute_fvd_lazy(
            data_fetch=data_fetch,
            T=args.T,
            num_samples=args.num_samples))
    # Save all metrics as a pickle file
    for mode in args.modes:
        metrics_pkl[mode] = new_metrics[mode]
    pickle.dump(metrics_pkl, open(pickle_path, "wb"))
    print(f"Saved metrics to {pickle_path}.")
