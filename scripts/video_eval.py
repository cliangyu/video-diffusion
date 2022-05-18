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
from collections import defaultdict
import tensorflow.compat.v1 as tf

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips as lpips_metric
from improved_diffusion.image_datasets import get_test_dataset
import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion import test_util

tf.disable_eager_execution() # Required for our FVD computation code


class LazyDataFetch:
    def __init__(self, dataset, eval_dir, obs_length, dataset_drange, drop_obs=True, num_samples=None):
        """ A class to handle loading sampled videos and their corresponding gt videos from the dataset.

        Arguments:
            drop_obs: if True, drops the observed part of the videos from the output pairs of videos.
            num_samples: if not None, asserts that all the videos have at least num_samples generated samples.
        """
        self.obs_length = obs_length
        self.drop_obs = drop_obs
        samples_dir = Path(eval_dir) / "samples"
        assert samples_dir.exists(), f"Samples dir {samples_dir} does not exist."
        filenames = [(x, [int(num) for num in x.stem.split("_")[-1].split("-")]) for x in samples_dir.glob("sample_*.npy")]
        # filenames has the following structure: [(filename, (video_idx, sample_idx))]
        filenames.sort(key=lambda x: x[1][0])
        # Arrange all the filenames in a dictionary from the test set index to a list of filenames (one filename for each generated sample).
        self.filenames_dict = defaultdict(list)
        for item in filenames:
            filename, (video_idx, sample_idx) = item
            self.filenames_dict[video_idx].append(filename)
        if num_samples is not None:
            for idx, filenames in self.filenames_dict.items():
                assert len(filenames) >= num_samples, f"Expected at least {num_samples} samples for each video, but found {len(filenames)} for video #{idx}"
        self.keys = list(self.filenames_dict.keys())
        self.dataset = dataset
        self.dataset_drange = dataset_drange
        assert self.dataset_drange[1] > self.dataset_drange[0]

    def __getitem__(self, idx):
        # Returns a tuple of (gt video, [list of sampled videos])
        # Each video has shape of TxCx3xHxW
        video_idx = self.keys[idx]
        filename_list = self.filenames_dict[video_idx]
        preds = {str(filename): (np.load(filename) / 255.0).astype(np.float32) for filename in filename_list} # pred with pixel values in [0, 1]
        gt = self.dataset[video_idx][0].numpy()
        gt = (gt - self.dataset_drange[0]) / (self.dataset_drange[1] - self.dataset_drange[0]) # gt with pixel values in [0, 1]
        gt = gt.astype(np.float32)
        if self.drop_obs:
            gt = gt[self.obs_length:]
            preds = {k: x[self.obs_length:] for k, x in preds.items()}
        return {"gt":gt,
                "preds": preds}

    def __len__(self):
        return len(self.keys)

    def get_num_samples(self):
        # Returns the number of samples per test video in the database. Assumes all test videos have the same number of samples
        return len(self[0]["preds"])

    @property
    def T(self):
        res = list(self[0]["preds"].values())[0].shape[0]
        if self.drop_obs:
            res += self.obs_length
        return res


class FVD:
    def __init__(self, batch_size, T, frame_shape):
        self.batch_size = batch_size
        self.vid = tf.placeholder("uint8", [self.batch_size, T, *frame_shape])
        self.vid_feature_vec = fvd.create_id3_embedding(fvd.preprocess(self.vid, (224, 224)), batch_size=self.batch_size)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

    def extract_features(self, vid):
        def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
            # From here: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            pad_size = target_length - array.shape[axis]
            if pad_size <= 0:
                return array
            npad = [(0, 0)] * array.ndim
            npad[axis] = (0, pad_size)
            return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

        # vid is expected to have a shape of BxTxCxHxW
        B = vid.shape[0]
        vid = np.moveaxis(vid, 2, 4) # B, T, H, W, C
        # Pad array, if required
        vid = pad_along_axis(vid, target_length=self.batch_size, axis=0)
        # Run the videos through the feature extractor and get the features.
        features = self.sess.run(self.vid_feature_vec, feed_dict={self.vid: vid})
        # Drop the paddings, if any
        features = features[:B]
        return features

    @staticmethod
    def compute_fvd(vid1_features, vid2_features):
        return fvd.fid_features_to_metric(vid1_features, vid2_features)

    @staticmethod
    def compute_kvd(vid1_features, vid2_features):
        # no batch limitations here, can pass in entire dataset
        print("WARNING: KID's subset size must be chosen more carefully. Currently using the number of all samples.")
        return fvd.kid_features_to_metric(vid1_features, vid2_features, kid_subset_size=len(vid1_features))

    # Pad arrays, if required


def compute_fvd_lazy(data_fetch, T, num_samples, batch_size=16):
    C, H, W = data_fetch[0]["gt"][0].shape
    fvd_handler = FVD(batch_size=batch_size, T=T, frame_shape=[H, W, C])
    with tf.Graph().as_default():
        num_videos = len(data_fetch)
        gt_features = np.zeros((num_videos, 400))
        pred_features = np.zeros((num_samples, num_videos, 400))
        fvd = np.zeros((num_samples))
        for i in tqdm(range(0, num_videos, batch_size), desc="FVD"):
            data_idx_min = i
            data_idx_max = min(i + batch_size, num_videos)
            data = [data_fetch[j] for j in range(data_idx_min, data_idx_max)]
            gt_batch = [item["gt"] for item in data]
            preds = [item["preds"] for item in data] # A list of size B. Each item is a dictionary from sample filenames to numpy arrays.
            preds = map(lambda x: list(zip(*x.items())), preds) # A list of size B. Each item is a tuple of (list of filenames, list of numpy arrays)
            preds_batch_names, preds_batch = list(zip(*list(preds)))
            gt_batch = np.stack(gt_batch)[:, :T] # BxTxCxHxW
            preds_batch = np.stack(preds_batch)[:, :num_samples, :T] # BxNxTxCxHxW
            # Cache filename: dir/.preds_batch_names-T.fvd_features.npz
            assert preds_batch.shape[1] == num_samples, f"Expected at least {num_samples} video prediction samples."
            # Convert image pixels to bytes
            gt_batch = (gt_batch * 255).astype("uint8")
            preds_batch = (preds_batch * 255).astype("uint8")

            # Compute features to be used in FVD computation
            if i == 0: print(gt_batch.shape)
            gt_f = fvd_handler.extract_features(gt_batch)
            gt_features[data_idx_min:data_idx_max] = gt_f
            # Compute features of the sampled videos
            for k in range(num_samples):
                pred_batch = preds_batch[:, k]
                if i == 0: print(pred_batch.shape)
                pred_f = fvd_handler.extract_features(pred_batch)
                # Update the overall features array
                pred_features[k, data_idx_min:data_idx_max] = pred_f
        for k in range(num_samples):
            fvd[k] = fvd_handler.compute_fvd(pred_features[k], gt_features)
    return {"fvd": fvd}


def compute_metrics_lazy(data_fetch, T, num_samples, C=3):
    T = T - data_fetch.obs_length
    num_videos = len(data_fetch)
    ssim = np.zeros((num_videos, num_samples, T))
    psnr = np.zeros((num_videos, num_samples, T))
    for i in tqdm(range(num_videos), desc="SSIM and PSNR"):
        data = data_fetch[i]
        gt = data["gt"]
        preds = list(data["preds"].values()) # Ignore the keys
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
    T = T - data_fetch.obs_length
    num_videos = len(data_fetch)
    lpips = np.zeros((num_videos, num_samples, T))
    loss_fn = lpips_metric.LPIPS(net='alex', spatial=False).to(device)
    for i in tqdm(range(num_videos), desc="LPIPS"):
        data = data_fetch[i]
        gt = data["gt"]
        preds = list(data["preds"].values()) # Ignore the keys
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_partition", default="test", choices=["train", "test"])
    parser.add_argument("--modes", nargs="+", type=str, default=["all"],
                        choices=["ssim", "psnr", "lpips", "fvd", "all"])
    parser.add_argument("--obs_length", type=int, default=36,
                        help="Number of observed frames. Default is 36.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--T", type=int, default=None,
                        help="Video length. If not given, the same T as used for training will be used.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of generated samples per test video.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="(Only used for FVD) Batch size for extracting video features (extracted from the I3D model). Default is 16.")
    args = parser.parse_args()

    if "all" in args.modes:
        args.modes = ["ssim", "psnr", "lpips", "fvd"]
    if args.dataset is None or args.T is None:
        model_config_path = Path(args.eval_dir) / "model_config.json"
        assert model_config_path.exists(), f"Could not find model config at {model_config_path}"
        with open(model_config_path, "r") as f:
            model_config = json.load(f)
            if args.dataset is None:
                args.dataset = model_config["dataset"]
            if args.T is None:
                args.T = model_config["T"]
    if args.batch_size is None:
        if "mazes" in args.dataset:
            args.batch_size = 16
        elif "minerl" in args.dataset:
            args.batch_size = 8
        elif "carla" in args.dataset:
            args.batch_size = 4
    # Load dataset
    dataset = locals()[f"get_{args.dataset_partition}_dataset"](dataset_name=args.dataset) # Load the full-length videos. We'll use the first T frames for evaluation, however.
    drange = [-1, 1] # Range of dataset's pixel values
    data_fetch = LazyDataFetch(
        dataset=dataset,
        eval_dir=args.eval_dir,
        obs_length=args.obs_length,
        dataset_drange=drange,
        num_samples=args.num_samples)
    if args.num_samples is None:
        args.num_samples = data_fetch.get_num_samples()
    if args.T is None:
        args.T = data_fetch.T
    else:
        assert args.T <= data_fetch.T

    # Check if metrics have already been computed
    name = f"metrics_{len(data_fetch)}-{args.num_samples}-{args.T}"
    pickle_path = Path(args.eval_dir) / f"{name}.pkl"
    if pickle_path.exists():
        metrics_pkl = pickle.load(open(pickle_path, "rb"))
        args.modes = [mode for mode in args.modes if mode not in metrics_pkl]
    print(f"Modes: {args.modes}")
    if len(args.modes) == 0:
        print("No metrics to compute.")
        quit(0)

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
        data_fetch_with_obs = LazyDataFetch(
            dataset=dataset,
            eval_dir=args.eval_dir,
            obs_length=args.obs_length,
            dataset_drange=drange,
            num_samples=args.num_samples,
            drop_obs=False)
        new_metrics.update(compute_fvd_lazy(
            data_fetch=data_fetch_with_obs,
            T=args.T,
            num_samples=args.num_samples,
            batch_size=args.batch_size))

    # Save all metrics as a pickle file (update it if it already exists)
    with test_util.Protect(pickle_path): # avoids race conditions
        if pickle_path.exists():
            metrics_pkl = pickle.load(open(pickle_path, "rb"))
        else:
            metrics_pkl = {}
        for mode in args.modes:
            metrics_pkl[mode] = new_metrics[mode]
        pickle.dump(metrics_pkl, open(pickle_path, "wb"))

    print(f"Saved metrics to {pickle_path}.")
