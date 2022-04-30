from ast import parse
from shutil import move
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import os
from tqdm.auto import tqdm
from pathlib import Path
import json

from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import dist_util
from improved_diffusion.image_datasets import get_test_dataset
from improved_diffusion.script_util import str2bool
from improved_diffusion import inference_util
from improved_diffusion import test_util


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
def infer_video(mode, model, diffusion, batch, max_frames, obs_length,
                step_size=1):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    B, T, C, H, W = batch.shape
    samples = torch.zeros_like(batch).cpu()
    samples[:, :obs_length] = batch[:, :obs_length]
    frame_indices_iterator = inference_util.inference_strategies[mode](video_length=T, num_obs=obs_length, max_frames=max_frames, step_size=step_size)

    for obs_frame_indices, latent_frame_indices in tqdm(frame_indices_iterator):
        print(f"Conditioning on {sorted(obs_frame_indices)} frames, predicting {sorted(latent_frame_indices)}.\n")
        # Prepare network's input
        x0 = torch.cat([samples[:, obs_frame_indices], samples[:, latent_frame_indices]], dim=1).clone()
        frame_indices = torch.cat([torch.tensor(obs_frame_indices), torch.tensor(latent_frame_indices)], dim=0).repeat((B, 1))
        # Prepare masks
        obs_mask, latent_mask, kinda_marg_mask = get_masks(x0, len(obs_frame_indices))
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
        samples[:, latent_frame_indices] = local_samples[:, -len(latent_frame_indices):].cpu()
    return samples.numpy()


def main(args, model, diffusion, dataloader, postfix="", dataset_indices=None):
    dataset_idx_translate = lambda idx: idx if dataset_indices is None else dataset_indices[idx]
    # Generate and store samples
    cnt = 0
    for batch, _ in tqdm(dataloader, leave=True):
        batch_size = len(batch)
        for sample_idx in range(args.num_samples) if args.sample_idx is None else [args.sample_idx]:
            output_filenames = [args.out_dir / f"sample_{dataset_idx_translate(cnt + i):04d}-{sample_idx}.npy" for i in range(batch_size)]
            todo = [not p.exists() for (i, p) in enumerate(output_filenames)] # Whether the file should be generated
            if not any(todo):
                print(f"Nothing to do for the batches {cnt} - {cnt + batch_size - 1}, sample #{sample_idx}.")
            else:
                if args.T is not None:
                    batch = batch[:, :args.T]
                batch = batch.to(args.device)
                recon = infer_video(mode=args.inference_mode, model=model, diffusion=diffusion,
                                    batch=batch, max_frames=args.max_frames, obs_length=args.obs_length,
                                    step_size=args.step_size)
                recon = (recon - drange[0]) / (drange[1] - drange[0])  * 255 # recon with pixel values in [0, 255]
                recon = recon.astype(np.uint8)
                for i in range(batch_size):
                    if todo[i]:
                        np.save(output_filenames[i], recon[i])
                        print(f"*** Saved {output_filenames[i]} ***")
                    else:
                        print(f"Skipped {output_filenames[i]}")
        cnt += batch_size


def visualise(args):
    vis = []
    frame_indices_iterator = inference_util.inference_strategies[args.inference_mode](
        video_length=args.T, num_obs=args.obs_length, max_frames=args.max_frames, step_size=args.step_size
    )
    exist_indices = list(range(args.obs_length))
    for obs_frame_indices, latent_frame_indices in tqdm(frame_indices_iterator):
        print(obs_frame_indices, latent_frame_indices)
        exist_indices.extend(latent_frame_indices)
        new_layer = torch.zeros((args.T, 3)).int()
        new_layer[exist_indices, 0] = 50
        new_layer[obs_frame_indices, 0] = 255
        new_layer[latent_frame_indices, 2] = 255
        vis.append(new_layer)
        vis.append(new_layer*0)
    from improved_diffusion.train_util import concat_images_with_padding
    from PIL import Image
    vis = torch.stack(vis)
    path = f"visualisations/sample_vis_{args.inference_mode}_T={args.T}_sampling_{args.step_size}_out_of_{args.max_frames}.png"
    Image.fromarray(vis.numpy().astype(np.uint8)).save(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default=None, help="Output directory for the generated videos. If None, defaults to a directory at samples/<checkpoint_dir_name>/<checkpoint_name>_<checkpoint_step>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference_mode", required=True, choices=inference_util.inference_strategies.keys())
    # Inference arguments
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of video frames (observed or latent) allowed to pass to the model at once. Defaults to what the model was trained with.")
    parser.add_argument("--obs_length", type=int, default=36,
                        help="Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest. Defaults to 36.")
    parser.add_argument("--step_size", type=int, default=1,
                        help="Number of frames to predict in each prediciton step. Defults to 1.")
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
    args = parser.parse_args()

    if args.just_visualise:
        visualise(args)
        exit()

    drange = [-1, 1] # Range of the generated samples' pixel values

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_step = data["step"]
    model_args.update({"use_ddim": args.use_ddim,
                       "timestep_respacing": args.timestep_respacing})
    # Update model parameters, if needed, to enable backward compatibility
    for k, v in default_model_configs.items():
        if k not in model_args:
            model_args[k] = v
    model_args = Namespace(**model_args)
    # Load the model
    model, diffusion = create_video_model_and_diffusion(
        **args_to_dict(model_args, video_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
            state_dict
        )
    model = model.to(args.device)
    model.eval()
    # Update max_frames if not set
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    print(f"max_frames = {args.max_frames}")
    # Load the test set
    dataset = get_test_dataset(dataset_name=model_args.dataset)
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
        dataset = torch.utils.data.Subset(dataset, list(range(args.subset_size)))
    elif args.indices is None:
        args.indices = list(range(len(dataset)))
        print(f"Generating predictions for the whole dataset.")
    # Take a subset of the dataset according to the indices
    dataset = torch.utils.data.Subset(dataset, args.indices)
    print(f"Dataset size (after subsampling according to indices) = {len(dataset)}")
    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # Prepare the diffusion sampling arguments (DDIM/respacing)
    postfix = ""
    if args.use_ddim:
        postfix += "_ddim"
    if args.timestep_respacing != "":
        postfix += "_" + f"respace{args.timestep_respacing}"

    # Create the output directory (if does not exist)
    if args.out_dir is None:
        name = f"{Path(args.checkpoint_path).stem}_{model_step}"
        if postfix != "":
            name += postfix
        args.out_dir = Path(f"samples/{Path(args.checkpoint_path).parent.name}/{name}")
    else:
        args.out_dir = Path(args.out_dir)
    args.out_dir = args.out_dir / f"{args.inference_mode}_{args.max_frames}_{args.step_size}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.out_dir}")

    # Store model configs in a JSON file
    json_path = args.out_dir / "model_config.json"
    if not json_path.exists():
        with test_util.Protect(json_path): # avoids race conditions
            with open(json_path, "w") as f:
                json.dump(vars(model_args), f, indent=4)
        print(f"Saved model config at {json_path}")

    # Generate the samples
    main(args, model, diffusion, dataloader, postfix=postfix, dataset_indices=args.indices)
