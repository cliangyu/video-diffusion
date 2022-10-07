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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args,
         models,
         diffusions,
         dataloader,
         postfix='',
         dataset_indices=None):
    assert (
        args.obs_length == 36
    ), 'Inference for observation lengths other than 36 is not implemented.'
    assert args.inference_mode == 'google'

    def dataset_idx_translate(idx):
        return idx if args.indices is None else args.indices[idx]

    cnt = 0
    for i, (batch, _) in enumerate(dataloader):
        print(f'Batch {i}')
        fnames = [
            args.eval_dir / 'elbos' /
            f'elbo_{dataset_idx_translate(cnt+j)}{postfix}.pkl'
            for j in range(len(batch))
        ]
        if all([os.path.exists(f) for f in fnames]):
            print('Already exist. Skipping', fnames)
            cnt += len(batch)
            continue
        returns = []
        inference_strategy = inference_strategies[args.inference_mode](
            video_length=args.T, num_obs=args.obs_length)
        for obs_indices, lat_indices in inference_strategy:
            obs_indices = [obs_indices for _ in range(len(batch))]
            lat_indices = [lat_indices for _ in range(len(batch))]
            model = models[inference_strategy._active_iterator]
            diffusion = diffusions[inference_strategy._active_iterator]
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
    max_frames = len(obs_indices[0]) + len(
        lat_indices[0]
    )  # This assumes that items in obs_indices have the same length. Same for lat_indices.
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
    parser.add_argument(
        '--fs1_path',
        type=str,
        required=True,
        help='Path to the checkpoint of the frameskip-1 model',
    )
    parser.add_argument(
        '--fs4_path',
        type=str,
        required=True,
        help='Path to the checkpoint of the frameskip-6 model',
    )
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
    parser.add_argument('--inference_mode', default='google')
    parser.add_argument(
        '--obs_length',
        type=int,
        default=36,
        help=
        'Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest. Defaults to 36.',
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
    args = parser.parse_args()
    assert args.inference_mode == 'google'
    args.checkpoint_path = args.fs4_path
    args.max_frames = None
    args.step_size = None

    # Load the checkpoint (state dictionary and config)
    models = {}
    diffusions = {}
    model_args_dict = {}
    for model_name in ['fs1', 'fs4']:
        data = dist_util.load_state_dict(getattr(args, f'{model_name}_path'),
                                         map_location='cpu')
        state_dict = data['state_dict']
        model_args = data['config']
        model_args.update({
            'use_ddim': args.use_ddim,
            'timestep_respacing': args.timestep_respacing
        })
        model_args = argparse.Namespace(**model_args)
        model_args_dict[model_name] = model_args
        # Load the models
        model, diffusion = create_video_model_and_diffusion(
            **args_to_dict(model_args,
                           video_model_and_diffusion_defaults().keys()))
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        model.eval()
        models[model_name], diffusions[model_name] = model, diffusion
    # Assertions on model arguments
    assert (
        model_args_dict['fs1'].dataset == model_args_dict['fs4'].dataset
    ), f"Both models shoould be trained on the same dataset. Got {model_args_dict['fs1'].dataset} and {model_args_dict['fs4'].dataset}."
    assert (
        model_args_dict['fs1'].mask_distribution ==
        'differently-spaced-groups-no-marg'
    ), f"Unexpected mask distribution for the fs1 model: {model_args_dict['fs1'].mask_distribution}"
    assert (
        model_args_dict['fs4'].mask_distribution == 'linspace-0-60-16'
    ), f"Unexpected mask distribution for the fs4 model: {model_args_dict['fs4'].mask_distribution}"

    # set up output directory
    args.eval_dir = test_util.get_model_results_path(
        args) / test_util.get_eval_run_identifier(args)
    (args.eval_dir / 'elbos').mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / 'elbos'}")

    # Load the test set
    if args.T is None:
        args.T = default_T_dict[model_args.dataset]
    dataset = locals()[f'get_{args.dataset_partition}_dataset'](
        dataset_name=model_args.dataset)
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
        print('Generating predictions for the whole dataset.')
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
        args.indices_path = args.eval_dir / 'frame_indices.pt'

    # Prepare the diffusion sampling arguments (DDIM/respacing)
    postfix = ''
    if args.use_ddim:
        postfix += '_ddim'
    if args.timestep_respacing != '':
        postfix += '_' + f'respace{args.timestep_respacing}'

    # Generate the samples
    main(
        args,
        models,
        diffusions,
        dataloader,
        postfix=postfix,
        dataset_indices=args.indices,
    )
