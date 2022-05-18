from ast import parse
from shutil import move
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import os, sys
from tqdm.auto import tqdm
from pathlib import Path
import json
from PIL import Image

from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import dist_util
from improved_diffusion.image_datasets import get_train_dataset, get_test_dataset
from improved_diffusion.script_util import str2bool
from improved_diffusion import inference_util
from improved_diffusion import test_util

sys.path.insert(1, str(Path(__file__).parent.resolve()))
from video_sample import visualise


# A dictionary of default model configs for the parameters newly introduced.
# It enables backward compatibility
default_model_configs = {"enforce_position_invariance": False,
                         "cond_emb_type": "channel"}


def get_masks(x0, num_obs):
    """ Generates observation, latent, and kinda-marginal masks.
    Assumes that the first num_obs frames are observed. and the rest are latent.
    Also, assumes that no frames are kinda-marginal.

    Args:
        x0 (torch.FloatTesnor): The input video.
        num_obs (int): Number of observed frames.

    Returns:
        Observation, latent, and kina-marginal masks
    """
    obs_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    obs_mask[:, :num_obs] = 1
    latent_mask = 1 - obs_mask
    kinda_marg_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    return obs_mask, latent_mask, kinda_marg_mask


@torch.no_grad()
def infer_video(mode, models, diffusions, batch, obs_length):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    B, T, C, H, W = batch.shape
    samples = torch.zeros_like(batch).cpu()
    samples[:, :obs_length] = batch[:, :obs_length]

    assert obs_length == 36, "Inference for observation lengths other than 36 is not implemented."
    assert mode == "google"
    inference_strategy = inference_util.inference_strategies[mode](
        video_length=T, num_obs=obs_length)

    for obs_frame_indices, latent_frame_indices in inference_strategy:
        print(f"Conditioning on {sorted(obs_frame_indices)} frames, predicting {sorted(latent_frame_indices)}.\n")
        model = models[inference_strategy._active_iterator]
        diffusion = diffusions[inference_strategy._active_iterator]
        # Prepare network's input
        x0 = torch.cat([samples[:, obs_frame_indices], samples[:, latent_frame_indices]], dim=1).clone()
        frame_indices = torch.cat([torch.tensor(obs_frame_indices), torch.tensor(latent_frame_indices)], dim=0).repeat((B, 1))
        obs_mask, latent_mask, kinda_marg_mask = get_masks(x0, len(obs_frame_indices))
        n_latent = len(latent_frame_indices)
        # Prepare masks
        print(f"{'Frame indices':20}: {frame_indices[0].cpu().numpy()}.")
        print(f"{'Observation mask':20}: {obs_mask[0].cpu().int().numpy().squeeze()}")
        print(f"{'Latent mask':20}: {latent_mask[0].cpu().int().numpy().squeeze()}")
        print("-" * 40)
        # Move tensors to the correct device
        [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices] = [xyz.to(batch.device) for xyz in [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices]]
        # Run the network
        local_samples, attention_map = diffusion.p_sample_loop(
            model, x0.shape, clip_denoised=True,
            model_kwargs=dict(frame_indices=frame_indices,
                              x0=x0,
                              obs_mask=obs_mask,
                              latent_mask=latent_mask,
                              kinda_marg_mask=kinda_marg_mask),
            latent_mask=latent_mask,
            return_attn_weights=False)
        # Fill in the generated frames
        if 'adaptive' in mode:
            n_obs = len(obs_frame_indices[0])
            for i, li in enumerate(latent_frame_indices):
                samples[i, li] = local_samples[i, n_obs:].cpu()
        else:
            samples[:, latent_frame_indices] = local_samples[:, -n_latent:].cpu()
    return samples.numpy()


def main(args, models, diffusions, dataloader, dataset_indices=None):
    dataset_idx_translate = lambda idx: idx if dataset_indices is None else dataset_indices[idx]
    # Generate and store samples
    cnt = 0
    for batch, _ in tqdm(dataloader, leave=True):
        batch_size = len(batch)
        for sample_idx in range(args.num_samples) if args.sample_idx is None else [args.sample_idx]:
            output_filenames = [args.eval_dir / "samples" / f"sample_{dataset_idx_translate(cnt + i):04d}-{sample_idx}.npy" for i in range(batch_size)]
            todo = [not p.exists() for (i, p) in enumerate(output_filenames)] # Whether the file should be generated
            if not any(todo):
                print(f"Nothing to do for the batches {cnt} - {cnt + batch_size - 1}, sample #{sample_idx}.")
            else:
                if args.T is not None:
                    batch = batch[:, :args.T]
                batch = batch.to(args.device)
                recon = infer_video(mode=args.inference_mode, models=models, diffusions=diffusions,
                                    batch=batch, obs_length=args.obs_length)
                recon = (recon - drange[0]) / (drange[1] - drange[0])  * 255 # recon with pixel values in [0, 255]
                recon = recon.astype(np.uint8)
                for i in range(batch_size):
                    if todo[i]:
                        np.save(output_filenames[i], recon[i])
                        print(f"*** Saved {output_filenames[i]} ***")
                    else:
                        print(f"Skipped {output_filenames[i]}")
        cnt += batch_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fs1_path", type=str, required=True,
                        help="Path to the checkpoint of the frameskip-1 model")
    parser.add_argument("--fs4_path", type=str, required=True,
                        help="Path to the checkpoint of the frameskip-6 model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", default=None, help="Path to the evaluation directory for the given checkpoint. If None, defaults to resutls/<checkpoint_dir_subset>/<checkpoint_name>.")
    parser.add_argument("--dataset_partition", default="test", choices=["train", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Inference arguments
    parser.add_argument("--inference_mode", default="google", choices=inference_util.inference_strategies.keys())
    parser.add_argument("--obs_length", type=int, default=36,
                        help="Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest. Defaults to 36.")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="If not None, only generate videos for the specified indices. Used for handling parallelization.")
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--T", type=int, default=None,
                        help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="If not None, only use a subset of the dataset. Defaults to the whole dataset.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate for each test video.")
    parser.add_argument("--sample_idx", type=int, default=None, help="Sampled images will have this specific index. Used for parallel sampling on multiple machines. If this argument is given, --num_samples is ignored.")
    parser.add_argument("--just_visualise", action='store_true', help="Make visualisation of sampling mode instead of doing it.")
    parser.add_argument("--big_visualise", action='store_true', help="Make visualisation big.")
    args = parser.parse_args()
    assert args.inference_mode == "google"
    args.checkpoint_path = args.fs4_path
    args.max_frames = None
    args.step_size = None

    if args.just_visualise:
        assert args.T is not None
        visualise(args)
        exit()

    drange = [-1, 1] # Range of the generated samples' pixel values

    # Load the checkpoints (state dictionary and config)
    models = {}
    diffusions = {}
    model_args_dict = {}
    for model_name in ["fs1", "fs4"]:
        data = dist_util.load_state_dict(getattr(args, f"{model_name}_path"), map_location="cpu")
        state_dict = data["state_dict"]
        model_args = data["config"]
        model_args.update({"use_ddim": args.use_ddim,
                           "timestep_respacing": args.timestep_respacing})
        model_args = Namespace(**model_args)
        model_args_dict[model_name] = model_args
        # Load the models
        model, diffusion = create_video_model_and_diffusion(
            **args_to_dict(model_args, video_model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
                state_dict
            )
        model = model.to(args.device)
        model.eval()
        models[model_name], diffusions[model_name] = model, diffusion
    # Assertions on model arguments
    assert model_args_dict["fs1"].dataset == model_args_dict["fs4"].dataset, f"Both models shoould be trained on the same dataset. Got {model_args_dict['fs1'].dataset} and {model_args_dict['fs4'].dataset}."
    assert model_args_dict["fs1"].mask_distribution == "differently-spaced-groups-no-marg", f"Unexpected mask distribution for the fs1 model: {model_args_dict['fs1'].mask_distribution}"
    assert model_args_dict["fs4"].mask_distribution == "linspace-0-60-16", f"Unexpected mask distribution for the fs4 model: {model_args_dict['fs4'].mask_distribution}"

    # Load the dataset
    dataset = locals()[f"get_{args.dataset_partition}_dataset"](dataset_name=model_args.dataset, T=args.T)
    print(f"Dataset size = {len(dataset)}")
    # Prepare the indices
    if args.indices is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        assert args.subset_size is None
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        args.indices = list(range(task_id * args.batch_size, (task_id + 1) * args.batch_size))
        print(f"Only generating predictions for the batch #{task_id}.")
    elif args.subset_size is not None:
        #indices = np.random.RandomState(123).choice(len(dataset), args.subset_size, replace=False)
        print(f"Only generating predictions for the first {args.subset_size} videos of the dataset.")
        args.indices = list(range(args.subset_size))
    elif args.indices is None:
        args.indices = list(range(len(dataset)))
        print(f"Generating predictions for the whole dataset.")
    # Take a subset of the dataset according to the indices
    dataset = torch.utils.data.Subset(dataset, args.indices)
    print(f"Dataset size (after subsampling according to indices) = {len(dataset)}")
    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    args.eval_dir = test_util.get_model_results_path(args) / test_util.get_eval_run_identifier(args)
    args.eval_dir = args.eval_dir
    (args.eval_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / 'samples'}")

    # Store model configs in a JSON file (only save the config of one of the models)
    for model_name in ["fs1", "fs4"]:
        json_path = args.eval_dir / ("model_config.json" if model_name == "fs4" else "model_config_fs1.json")
        model_args = model_args_dict[model_name]
        if not json_path.exists():
            with test_util.Protect(json_path): # avoids race conditions
                with open(json_path, "w") as f:
                    json.dump(vars(model_args), f, indent=4)
            print(f"Saved model config at {json_path}")

    # Generate the samples
    main(args, models, diffusions, dataloader, dataset_indices=args.indices)
