"""Approximate the bits/dimension for an image model."""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from video_sample import default_model_configs, get_masks

from improved_diffusion import dist_util, logger, test_util
from improved_diffusion.image_datasets import (default_T_dict,
                                               get_test_dataset,
                                               get_train_dataset, load_data)
from improved_diffusion.inference_util import inference_strategies
from improved_diffusion.script_util import (args_to_dict,
                                            create_video_model_and_diffusion,
                                            str2bool,
                                            video_model_and_diffusion_defaults)

mask_distributions = ['differently-spaced-groups']


def get_eval_frame_indices(args, batch=None, optimal_schedule_path=None):
    """can get indices by either.

    - distribution from inference_utils
    - loading from directory
    """
    if args.inference_mode not in inference_strategies.keys():
        obs_indices, lat_indices = torch.load(args.indices_path)
        print('loaded inference frame indices')
    else:
        adaptive_kwargs = dict(distance='lpips') if args.adaptive else {}
        frame_indices_iterator = inference_strategies[args.inference_mode](
            video_length=args.T,
            num_obs=args.obs_length,
            max_frames=args.max_frames,
            step_size=args.step_size,
            optimal_schedule_path=optimal_schedule_path,
            **adaptive_kwargs,
        )
        if args.adaptive:
            frame_indices_iterator.set_videos(batch)
        obs_lat_indices = list(frame_indices_iterator)
        obs_indices = [pair[0]
                       for pair in obs_lat_indices]  # steps by batch_elements
        lat_indices = [pair[1] for pair in obs_lat_indices]  # steps
        if (
                args.adaptive
        ):  # want obs_indices, lat_indices to both be batch_elements x steps x index_sequence
            obs_indices = [[
                obs_indices[i][j] for i in range(len(obs_indices))
            ] for j in range(len(batch))]
            lat_indices = [[
                lat_indices[i][j] for i in range(len(lat_indices))
            ] for j in range(len(batch))]
        else:
            obs_indices = [obs_indices for _ in range(args.test_set_size)]
            lat_indices = [lat_indices for _ in range(args.test_set_size)]
        print('generated inference frame indices')
        if os.path.exists(args.indices_path) and not args.adaptive:
            print(f'Checking match to indices at {args.indices_path}')
            try:
                obs_to_check, lat_to_check = torch.load(args.indices_path)
            except EOFError:  # possibly an issue with parallelism that we can solve by waiting for a while
                time.sleep(5)
                obs_to_check, lat_to_check = torch.load(args.indices_path)
            for i1, i2 in zip(obs_indices, obs_to_check):
                assert i1 == i2
            for i1, i2 in zip(lat_indices, lat_to_check):
                assert i1 == i2
        elif not args.adaptive:
            torch.save((obs_indices, lat_indices), args.indices_path)
    return obs_indices, lat_indices


def main(args, model, diffusion, dataloader, postfix='', dataset_indices=None):
    optimal_schedule_path = (None if args.optimality is None else
                             args.eval_dir / 'optimal_schedule.pt')

    def dataset_idx_translate(idx):
        return idx if args.indices is None else args.indices[idx]

    cnt = 0
    for i, (batch, _) in enumerate(dataloader):
        fnames = [
            args.eval_dir / 'elbos' /
            f'elbo_{dataset_idx_translate(cnt+j)}{postfix}.pkl'
            for j in range(len(batch))
        ]
        if all([os.path.exists(f) for f in fnames]):
            print('Already exist. Skipping', fnames)
            cnt += len(batch)
            continue
        obs_indices, lat_indices = get_eval_frame_indices(
            args,
            batch=batch if args.adaptive else None,
            optimal_schedule_path=optimal_schedule_path,
        )
        if not args.adaptive:
            # frame_indices = ([obs_indices[i] for i in args.indices], [lat_indices[i] for i in args.indices],)
            obs_indices = [obs_indices[i] for i in args.indices]
            lat_indices = [lat_indices[i] for i in args.indices]
        batch_obs_indices = (obs_indices if args.adaptive else
                             obs_indices[cnt:cnt + len(batch)])
        batch_lat_indices = (lat_indices if args.adaptive else
                             lat_indices[cnt:cnt + len(batch)])
        returns = []
        n_index_types = len(batch_obs_indices[0])
        for i in range(n_index_types):
            obs_indices = [b[i] for b in batch_obs_indices]
            lat_indices = [b[i] for b in batch_lat_indices]
            returns.append(
                run_bpd_evaluation(
                    model=model,
                    diffusion=diffusion,
                    batch=batch,
                    clip_denoised=args.clip_denoised,
                    obs_indices=obs_indices,
                    lat_indices=lat_indices,
                ))
        returns = {
            k: np.stack([r[k] for r in returns], axis=1)
            for k in returns[0].keys()
        }
        for j in range(len(returns['total_bpd'])):
            fname = fnames[j]
            pickle.dump({k: v[j]
                         for k, v in returns.items()}, open(fname, 'wb'))
            print('Saved to', fname)
        cnt += len(batch)


def run_bpd_evaluation(model,
                       diffusion,
                       batch,
                       clip_denoised,
                       obs_indices,
                       lat_indices,
                       t_seq=None):
    max_frames = max(
        len(o) + len(l) for o, l in zip(obs_indices, lat_indices)
    )  # len(obs_indices[0]) + len(lat_indices[0]) didn't work for variable length obs/lat indices
    x0 = torch.zeros_like(batch[:, :max_frames].to(dist_util.dev()))
    obs_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    lat_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    kinda_marg_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    frame_indices = torch.zeros_like(x0[:, :, 0, 0, 0]).long()
    for i, (obs_i, lat_i) in enumerate(zip(obs_indices, lat_indices)):
        x0[i, :len(obs_i)] = batch[i, obs_i]
        obs_mask[i, :len(obs_i)] = 1.0
        frame_indices[i, :len(obs_i)] = torch.tensor(obs_i)
        x0[i, len(obs_i):len(obs_i) + len(lat_i)] = batch[i, lat_i]
        lat_mask[i, len(obs_i):len(obs_i) + len(lat_i)] = 1.0
        frame_indices[i,
                      len(obs_i):len(obs_i) + len(lat_i)] = torch.tensor(lat_i)
    model_kwargs = dict(
        frame_indices=frame_indices,
        x0=x0,
        obs_mask=obs_mask,
        latent_mask=lat_mask,
        kinda_marg_mask=kinda_marg_mask,
    )
    metrics = diffusion.calc_bpd_loop_subsampled(
        model,
        x0,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        latent_mask=lat_mask,
        t_seq=t_seq,
    )

    metrics = {
        k: v.sum(dim=1) if v.ndim > 1 else v
        for k, v in metrics.items()
    }
    # sum (rather than mean) over frame dimension by multiplying by number of frames
    metrics = {k: v * max_frames for (k, v) in metrics.items()}
    metrics = {k: v.detach().cpu().numpy() for k, v in metrics.items()}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--eval_dir',
        help=
        'Ideally set to samples/<checkpoint_id>/. Will store in subdirectory corresponding to inference mode.',
    )
    parser.add_argument('--dataset_partition',
                        default='test',
                        choices=['train', 'test'])
    parser.add_argument('--indices_path', type=str, default=None)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    # Inference arguments
    parser.add_argument('--inference_mode', required=True)
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help=
        'Maximum number of video frames (observed or latent) allowed to pass to the model at once. Defaults to what the model was trained with.',
    )
    parser.add_argument(
        '--obs_length',
        type=int,
        default=36,
        help=
        'Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest. Defaults to 36.',
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=1,
        help=
        'Number of frames to predict in each prediciton step. Defults to 1.',
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='*',
        default=None,
        help=
        'If not None, only generate videos for the specified indices. Used for handling parallelization.',
    )
    parser.add_argument('--use_ddim', type=str2bool, default=False)
    parser.add_argument('--timestep_respacing', type=str, default='')
    parser.add_argument(
        '--T',
        type=int,
        default=None,
        help=
        'Length of the videos. If not specified, it will be inferred from the dataset.',
    )
    parser.add_argument(
        '--clip_denoised',
        type=str2bool,
        default=True,
        help='Clip model predictions of x0 to be in valid range.',
    )
    parser.add_argument(
        '--optimality',
        type=str,
        default=None,
        choices=[
            'linspace-t',
            'random-t',
            'linspace-t-force-nearby',
            'random-t-force-nearby',
        ],
        help=
        'Whcih optimality schedule to use for choosing observed frames. The optimal schedule should be generated before via video_optimal_schedule.py. Default is to not use any optimality.',
    )
    args = parser.parse_args()
    args.adaptive = 'adaptive' in args.inference_mode

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location='cpu')
    state_dict = data['state_dict']
    model_args = data['config']
    model_step = data['step']
    model_args.update({
        'use_ddim': args.use_ddim,
        'timestep_respacing': args.timestep_respacing
    })
    # Update model parameters, if needed, to enable backward compatibility
    for k, v in default_model_configs.items():
        if k not in model_args:
            model_args[k] = v
    model_args = argparse.Namespace(**model_args)
    # Load the model
    model, diffusion = create_video_model_and_diffusion(
        **args_to_dict(model_args,
                       video_model_and_diffusion_defaults().keys()))
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    # Update max_frames if not set
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    print(f'max_frames = {args.max_frames}')

    # set up output directory
    args.eval_dir = test_util.get_model_results_path(
        args) / test_util.get_eval_run_identifier(args)
    (args.eval_dir / 'elbos').mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / 'elbos'}")

    # Load the test set
    if args.T is None:
        args.T = default_T_dict[model_args.dataset]
    dataset = locals()[f'get_{args.dataset_partition}_dataset'](
        dataset_name=model_args.dataset, T=args.T)
    args.test_set_size = len(dataset)
    print(f'Dataset size = {args.test_set_size}')
    # Prepare the indices
    if args.indices is None and 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        args.indices = list(
            range(task_id * args.batch_size, (task_id + 1) * args.batch_size))
        print(f'Only generating predictions for the batch #{task_id}.')
    elif args.indices is None:
        args.indices = list(range(len(dataset)))
        print(f'Generating predictions for the whole dataset.')
    else:
        raise NotImplementedError
    # Take a subset of the dataset according to the indices
    dataset = torch.utils.data.Subset(dataset, args.indices)
    print(
        f'Dataset size (after subsampling according to indices) = {len(dataset)}'
    )
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False)
    if args.indices_path is None:
        args.indices_path = args.eval_dir / f'frame_indices.pt'

    # Prepare the diffusion sampling arguments (DDIM/respacing)
    postfix = ''
    if args.use_ddim:
        postfix += '_ddim'
    if args.timestep_respacing != '':
        postfix += '_' + f'respace{args.timestep_respacing}'

    # Store model configs in a JSON file
    json_path = args.eval_dir / 'model_config.json'
    if not json_path.exists():
        with test_util.Protect(json_path):  # avoids race conditions
            with open(json_path, 'w') as f:
                json.dump(vars(model_args), f, indent=4)
        print(f'Saved model config at {json_path}')

    # Generate the samples
    main(
        args,
        model,
        diffusion,
        dataloader,
        postfix=postfix,
        dataset_indices=args.indices,
    )
