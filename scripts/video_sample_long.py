import json
import os
import shutil
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from improved_diffusion import dist_util, inference_util, test_util
from improved_diffusion.image_datasets import (get_test_dataset,
                                               get_train_dataset,
                                               get_variable_length_dataset)
from improved_diffusion.script_util import str2bool

# A dictionary of default model configs for the parameters newly introduced.
# It enables backward compatibility
default_model_configs = {
    'enforce_position_invariance': False,
    'cond_emb_type': 'channel',
}


def get_masks(x0, num_obs):
    """Generates observation, latent, and kinda-marginal masks. Assumes that
    the first num_obs frames are observed. and the rest are latent. Also,
    assumes that no frames are kinda-marginal.

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
def infer_step(model, diffusion, video, args):
    drange = [-1, 1]  # Range of the generated samples' pixel values
    video = (video / 255.0) * (drange[1] -
                               drange[0]) - 1  # Normalize to [-1, 1]
    video = video[:args.obs_length]
    video = torch.tensor(video)
    assert (len(video) == args.obs_length
            ), f'Expected {args.obs_length} frames, but got {len(video)}'
    T, C, H, W = video.shape
    samples = torch.zeros(1, args.obs_length + args.file_length, C, H, W)
    samples[:, :args.obs_length] = video[:args.obs_length]

    adaptive_kwargs = (dict(
        distance='lpips') if 'adaptive' in args.inference_mode else {})
    frame_indices_iterator = inference_util.inference_strategies[
        args.inference_mode](
            video_length=samples.shape[1],
            num_obs=T,
            max_frames=args.max_frames,
            step_size=args.step_size,
            **adaptive_kwargs,
        )

    while True:
        if 'adaptive' in args.inference_mode:
            frame_indices_iterator.set_videos(samples.to(args.device))
        try:
            obs_indices, lat_indices = next(frame_indices_iterator)
        except StopIteration:
            break
        if 'adaptive' in args.inference_mode:
            frame_indices = torch.cat(
                [torch.tensor(obs_indices),
                 torch.tensor(lat_indices)], dim=1).long()
            x0 = torch.stack(
                [samples[i, fi] for i, fi in enumerate(frame_indices)],
                dim=0).clone()
            obs_mask, latent_mask, kinda_marg_mask = get_masks(
                x0, len(obs_indices[0]))
            n_latent = len(lat_indices[0])
        else:
            x0 = torch.cat([samples[:, obs_indices], samples[:, lat_indices]],
                           dim=1).clone()
            frame_indices = torch.cat(
                [torch.tensor(obs_indices),
                 torch.tensor(lat_indices)], dim=0).repeat((1, 1))
            # Prepare masks
            obs_mask, latent_mask, kinda_marg_mask = get_masks(
                x0, len(obs_indices))
            n_latent = len(lat_indices)
        # Move tensors to the correct device
        [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices] = [
            xyz.to(args.device) for xyz in
            [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices]
        ]
        print(f"{'Frame indices':20}: {frame_indices[0].cpu().numpy()}.")
        print(
            f"{'Observation mask':20}: {obs_mask[0].cpu().int().numpy().squeeze()}"
        )
        print(
            f"{'Latent mask':20}: {latent_mask[0].cpu().int().numpy().squeeze()}"
        )
        print('-' * 40)
        # Run the network
        local_samples, attention_map = diffusion.p_sample_loop(
            model,
            x0.shape,
            clip_denoised=True,
            model_kwargs=dict(
                frame_indices=frame_indices,
                x0=x0,
                obs_mask=obs_mask,
                latent_mask=latent_mask,
                kinda_marg_mask=kinda_marg_mask,
            ),
            latent_mask=latent_mask,
            return_attn_weights=False,
            use_gradient_method=args.use_gradient_method,
        )
        # Fill in the generated frames
        samples[:, lat_indices] = local_samples[:, -n_latent:].cpu()
    # Drop the batch dimension and the first few observed frames
    samples = samples[0][args.obs_length:]
    # Tranform pixel values to [0,255]
    samples = ((samples - drange[0]) / (drange[1] - drange[0]) * 255
               )  # samples with pixel values in [0, 255]
    return samples.numpy()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument(
        '--starting_video',
        type=str,
        default=None,
        help=
        'Path to the video file to start generation with. It should be a .npy file. If None, starts from the latest sampled video stored at the output direcotyr',
    )
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument(
        '--inference_mode',
        type=str,
        required=True,
        choices=[
            'autoreg',
            'hierarchy-2',
            'mixed-autoreg-independent',
            'adaptive-hierarchy-2',
        ],
    )
    parser.add_argument(
        '--eval_dir',
        default=None,
        help=
        'Path to the evaluation directory for the given checkpoint. If None, defaults to resutls/<checkpoint_dir_subset>/<checkpoint_name>.',
    )
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-o', '--out', default=None)
    parser.add_argument('--unconditional', action='store_true')
    # Inference arguments
    parser.add_argument('--use_gradient_method', action='store_true')
    parser.add_argument('--use_ddim', type=str2bool, default=False)
    parser.add_argument('--timestep_respacing', type=str, default='')
    args = parser.parse_args()

    if args.out is None:
        args.out = f'videos/long/{time.strftime("%Y%m%d-%H%M%S")}'
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load the checkpoint (state dictionary and config)
    (model, diffusion), model_args = test_util.load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        use_ddim=args.use_ddim,
        timestep_respacing=args.timestep_respacing,
    )
    args.max_frames = model_args.max_frames
    args.step_size = args.max_frames // 2
    args.obs_length = model_args.T // 2
    # Files are going to have the same number of frames as the model's T
    args.file_length = model_args.T - args.obs_length
    assert (
        args.file_length >= args.obs_length
    ), f"File length ({args.file_length}) must be greater than observation length ({args.obs_length}). It's due to the limitation of this implementation."

    model_args.max_frames = args.max_frames
    model_args.step_size = args.step_size
    model_args.obs_length = args.obs_length
    model_args.file_length = args.file_length
    model_args.inference_mode = args.inference_mode

    config_path = args.out / 'config.json'
    if not any(args.out.iterdir()):
        # Output directory is empty
        # -> Make sure --starting_video argument is valid.
        # -> copy the starting video as the first video in the output directory.
        # -> save the model config in the directory.
        if not args.unconditional:
            assert (
                args.starting_video is not None
            ), '--starting_video argument is required when the output directory is empty.'
            args.starting_video = Path(args.starting_video)
            assert (args.starting_video.is_file()
                    ), f'Starting video {args.starting_video} does not exist.'
            shutil.copyfile(args.starting_video, args.out / 'video_0.npy')
        else:
            assert (
                args.starting_video is None
            ), f'--starting_video argument should be None for unconditional sampling (got {args.starting_video}).'
            cond_obs_length = args.obs_length
            args.obs_length = 0
        with open(config_path, 'w') as f:
            json.dump(vars(model_args), f, indent=4)
        print(f'Saved model config at {config_path}')
        video_index_offset = 1
    else:
        # Make sure the --starting_video argument is not given.
        # Make sure the model config matches the saved one.
        assert (
            args.starting_video is None
        ), '--starting_video argument is not allowed when the output directory is not empty.'
        args.starting_video = sorted(
            list(args.out.glob('video_*.npy')),
            key=lambda x: int(x.stem.split('_')[1]))[-1]
        args.starting_video = Path(args.starting_video)
        video_index_offset = int(args.starting_video.stem.split('_')[1]) + 1
        assert config_path.exists(
        ), f'Model config file {config_path} does not exist.'
        with open(config_path, 'r') as f:
            loaded_args = json.load(f)
        assert (
            vars(model_args) == loaded_args
        ), f'Model config does not match the one saved at {config_path}.'

    # Print run information
    print(
        f'Starting from {args.starting_video}, generating {args.length} frames, and storing the results at {args.out}.'
    )
    print(
        f'Each generated file will have {args.file_length} frames and is conditioned on the past {args.obs_length} frames.'
    )
    print(f'max_frames = {args.max_frames} and step_size = {args.step_size}')

    if not args.unconditional:
        # Load the video to start generation with.
        video = np.load(args.starting_video)
        assert (
            len(video) >= args.obs_length
        )  # Make sure the video is long enough to be able to condition on its last frames.
    else:
        res = 128 if 'carla' in model_args.dataset else 64
        video = np.zeros((0, 3, res, res), dtype=np.uint8)

    # Generate the video
    for cnt, frame_idx in enumerate(range(0, args.length, args.file_length)):
        path = args.out / f'video_{video_index_offset + cnt}.npy'
        assert (
            not path.exists()
        ), f'About to generate video #{video_index_offset + cnt} at {path} but this file already exists.'
        new_video = infer_step(model, diffusion, video, args)
        np.save(path, new_video)
        print(f'Saved a video part (with {len(new_video)} frames) at {path}')
        video = np.concatenate(
            [video, new_video], axis=0
        )[-args.
          obs_length:]  # Drop anything farther than obs_length frames since we are not conditioning on them.
        if args.unconditional and args.obs_length == 0:
            args.obs_length = cond_obs_length
