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
import sys

from improved_diffusion.image_datasets import get_test_dataset
import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion import test_util

sys.path.insert(1, str(Path(__file__).parent.resolve()))
from video_eval import LazyDataFetch


def compute_FVD_features(vid1, vid2): 
    # batch has to be == 16 in fvd.create_id3_embedding
    assert vid1.shape[0] == 16 
    assert vid2.shape[0] == 16

    # assert channel is last dimension
    # and trust that shape is B, T, H, W, C
    assert vid1.shape[4] == 3
    assert vid2.shape[4] == 3

    # assert pixel values are bytes
    assert vid1.dtype == np.uint8
    assert vid2.dtype == np.uint8
    
    with tf.Graph().as_default():
        vid1 = tf.convert_to_tensor(vid1, np.uint8)
        vid2 = tf.convert_to_tensor(vid2, np.uint8)
        
        vid1_feature_vec = fvd.create_id3_embedding(fvd.preprocess(vid1,(224, 224)))
        vid2_feature_vec = fvd.create_id3_embedding(fvd.preprocess(vid2,(224, 224)))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            return sess.run(vid1_feature_vec), sess.run(vid2_feature_vec)


def compute_FVD(vid1_features, vid2_features): 
    # no batch limitations here, can pass in entire dataset
    with tf.Graph().as_default():
        vid1_features = tf.convert_to_tensor(vid1_features, np.float32)
        vid2_features = tf.convert_to_tensor(vid2_features, np.float32)
        
        result = fvd.calculate_fvd(vid1_features, vid2_features)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            return sess.run(result)



def compute_fvd_lazy(data_fetch, T, num_samples):
    num_videos = len(data_fetch)
    gt_features = np.zeros((num_samples, num_videos, 400))
    pred_features = np.zeros((num_samples, num_videos, 400))
    fvd = np.zeros((num_samples))
    for i in tqdm(range(0, num_videos, 16)):
        data_idx_min = i
        data_idx_max = min(i + 16, num_videos)
        data = [data_fetch[j] for j in range(data_idx_min, data_idx_max)]
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
        # Pad arrays, if required
        def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
            # From here: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            pad_size = target_length - array.shape[axis]
            if pad_size <= 0:
                return array
            npad = [(0, 0)] * array.ndim
            npad[axis] = (0, pad_size)

            return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        gt_batch = pad_along_axis(gt_batch, target_length=16, axis=0)
        preds_batch = pad_along_axis(preds_batch, target_length=16, axis=0)
        # Compute features to be used in FVD computation
        for k in range(num_samples):
            pred_batch = preds_batch[:, k]
            print(pred_batch.shape, gt_batch.shape)
            pred_f, gt_f = compute_FVD_features(pred_batch, gt_batch)
            # Drop the paddings, if any
            pred_f = pred_f[:data_idx_max - data_idx_min]
            gt_f = gt_f[:data_idx_max - data_idx_min]
            # Update the overall features array
            pred_features[k, data_idx_min:data_idx_max] = pred_f
            gt_features[k, data_idx_min:data_idx_max] = gt_f
    for k in range(num_samples):
        fvd[k] = compute_FVD(pred_features[k], gt_features[k])
    return {"fvd": fvd}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--obs_length", type=int, required=True,
                        help="Number of observed frames.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--T", type=int, default=None,
                        help="Video length. If not given, the length of the dataset is used.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of generated samples per test video.")
    args = parser.parse_args()

    if args.dataset is None:
        model_config_path = Path(args.samples_dir) / "model_config.json"
        assert model_config_path.exists(), f"Could not find model config at {model_config_path}"
        with open(model_config_path, "r") as f:
            args.dataset = json.load(f)["dataset"]
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
        if "fvd" in metrics_pkl:
            print(f"FVD already computed for {args.samples_dir}")
            exit(0)
    else:
        metrics_pkl = {}

    # Compute metrics
    new_metrics = {}
    new_metrics.update(compute_fvd_lazy(
        data_fetch=data_fetch,
        T=args.T,
        num_samples=args.num_samples))
    # Save all metrics as a pickle file
    metrics_pkl["fvd"] = new_metrics["fvd"]

    with test_util.Protect(pickle_path): # avoids race conditions
        pickle.dump(metrics_pkl, open(pickle_path, "wb"))

    print(f"Saved metrics to {pickle_path}.")
