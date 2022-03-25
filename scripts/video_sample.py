from shutil import move
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import os
from tqdm.auto import tqdm
from pathlib import Path
import time

from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import dist_util
from improved_diffusion.image_datasets import get_test_dataset


@torch.no_grad()
def get_indices_simple_autoreg(cur_idx, n=1):
    """
    n (int, optional): How many frames to generate. Defaults to 1.
    """
    distances_past = torch.arange(n, cur_idx+1)
    obs_frame_indices = (cur_idx - distances_past)
    latent_frame_indices = cur_idx - torch.arange(0, n)
    return obs_frame_indices, latent_frame_indices


@torch.no_grad()
def get_indices_exp_past(cur_idx):
    distances_past = 2**torch.arange(int(np.log2(cur_idx))) # distances from the observed frames (all in the past)
    obs_frame_indices = (cur_idx - distances_past)
    latent_frame_indices = torch.tensor([cur_idx]).type(obs_frame_indices.type())
    return obs_frame_indices, latent_frame_indices


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
def infer_video(mode, model, diffusion, batch, max_T, obs_length, assert_fits_gpu=True):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    B, T, C, H, W = batch.shape
    samples = torch.zeros_like(batch).cpu()
    samples[:, :obs_length] = batch[:, :obs_length]
    if mode == "simple-autoreg":
        get_indices = get_indices_simple_autoreg
        get_indices_kwargs = dict(n=1)
    elif mode == "exp-past":
        get_indices = get_indices_exp_past
        get_indices_kwargs = {}
    elif mode == "multi-granuality":
        raise NotImplementedError(f"Inference mode {mode} is not implemented yet.")
    elif mode == "independent":
        raise NotImplementedError(f"Inference mode {mode} is not implemented yet.")
    else:
        raise NotImplementedError(f"Inference mode {mode} is invalid.")

    for i in tqdm(range(obs_length, T)):
        # Prepare frame indices
        obs_frame_indices, latent_frame_indices = get_indices(i, **get_indices_kwargs)
        if len(obs_frame_indices) > max_T - 1:
            print(f"**WARNING**: Dropping {len(obs_frame_indices) - max_T + 1} of the observed frames because they don't fir in the GPU memory.")
            obs_frame_indices = obs_frame_indices[:max_T-1] # drops the last frames, if not fitting in the GPU memory.
            if assert_fits_gpu:
                raise RuntimeError("The batch is too large to fit in the GPU memory.")
        print(f"Conditioning on {obs_frame_indices.numpy()} frames, predicting {latent_frame_indices.numpy()}.")
        # Prepare network's input
        x0 = torch.cat([samples[:, obs_frame_indices], samples[:, latent_frame_indices]], dim=1)
        frame_indices = torch.cat([obs_frame_indices, latent_frame_indices], dim=0).repeat((B, 1))
        # Prepare masks
        obs_mask, latent_mask, kinda_marg_mask = get_masks(x0, len(obs_frame_indices))
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


@torch.no_grad()
def dryrun_gpu_memory(args, model, diffusion, dataloader):
    import pynvml
    pynvml.nvmlInit()
    gpu_info_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    batch = next(iter(dataloader))[0].to(args.device)
    mode = args.inference_mode
    max_T = args.max_T
    assert_fits_gpu = args.assert_fits_gpu
    B, T, C, H, W = batch.shape
    samples_base = batch[:1].cpu().clone() # Start with a batch size of 1
    if mode == "simple-autoreg":
        get_indices = get_indices_simple_autoreg
        get_indices_kwargs = dict(n=1)
    elif mode == "exp-past":
        get_indices = get_indices_exp_past
        get_indices_kwargs = {}
    elif mode == "multi-granuality":
        raise NotImplementedError(f"Inference mode {mode} is not implemented yet.")
    elif mode == "independent":
        raise NotImplementedError(f"Inference mode {mode} is not implemented yet.")
    else:
        raise NotImplementedError(f"Inference mode {mode} is invalid.")

    i = T - 1
    # Prepare frame indices
    obs_frame_indices, latent_frame_indices = get_indices(i, **get_indices_kwargs)
    if len(obs_frame_indices) > max_T - 1:
            print(f"**WARNING**: Dropping {len(obs_frame_indices) - max_T + 1} of the observed frames because they don't fir in the GPU memory.")
            obs_frame_indices = obs_frame_indices[:max_T-1] # drops the last frames, if not fitting in the GPU memory.
            if assert_fits_gpu:
                raise RuntimeError("The batch is too large to fit in the GPU memory.")
    print(f"Conditioning on {obs_frame_indices.numpy()} frames, predicting {latent_frame_indices.numpy()}.")
    # Prepare network's input
    x0_base = torch.cat([samples_base[:, obs_frame_indices], samples_base[:, latent_frame_indices]], dim=1)
    frame_indices_base = torch.cat([obs_frame_indices, latent_frame_indices], dim=0).repeat((1, 1))
    # Prepare masks
    obs_mask_base, latent_mask_base, kinda_marg_mask_base = get_masks(x0_base, len(obs_frame_indices))
    sm = 0
    for B in range(1, 1000):
        # Fix the shapes
        x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices = [xyz.repeat_interleave(B, dim=0)
            for xyz in [x0_base, obs_mask_base, latent_mask_base, kinda_marg_mask_base, frame_indices_base]]
        # Move tensors to the correct device
        [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices] = [xyz.to(batch.device)
            for xyz in [x0, obs_mask, latent_mask, kinda_marg_mask, frame_indices]]
        x0 = torch.randn_like(x0)
        # Run the network
        print(f"Model input batch size (to double check): {len(x0)}")
        t_0 = time.time()
        local_samples, attention_map = diffusion.p_sample_loop(
            model, x0.shape, clip_denoised=True,
            model_kwargs=dict(frame_indices=frame_indices,
                              x0=x0,
                              obs_mask=obs_mask,
                              latent_mask=latent_mask,
                              kinda_marg_mask=kinda_marg_mask),
            latent_mask=latent_mask,
            return_attn_weights=False)
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_info_handle)
        gpu_mem_info = {k: getattr(info, k) for k in ["used", "free", "total"]}
        gpu_mem_info = {k: v / 2**20 for k, v in gpu_mem_info.items()}
        print("#" * 20)
        print(f"Passed for a batch size of {B}.")
        print(f"Took {time.time() - t_0:.2f}s.")
        print(f"GPU usage: {gpu_mem_info['used']:,.2f} / {gpu_mem_info['total']:,.2f} = {gpu_mem_info['used'] / gpu_mem_info['total'] * 100:.2f}%")
        print("#" * 20)
        sm += local_samples.sum().item() # Don't know if Python might optimize the code at runtime and ignore local_samples. This bit avoid the potnetial optimization.


def main(args, model, diffusion, dataloader):
    # Prepare the indices
    if args.indices is None:
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
            args.indices = list(range(task_id * args.batch_size, (task_id + 1) * args.batch_size))
        else:
            args.indices = list(range(len(dataset)))
    # Create the output directory (if does not exist)
    model_step = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")["step"]
    if args.out_dir is None:
        args.out_dir = Path(f"samples/{Path(args.checkpoint_path).parent.name}/{Path(args.checkpoint_path).stem}_{model_step}")
    else:
        args.out_dir = Path(args.out_dir)
    args.out_dir = args.out_dir / args.inference_mode
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.out_dir}")

    # Generate and store samples
    cnt = 0
    for batch, _ in tqdm(dataloader, leave=True):
        batch_size = len(batch)
        output_filenames = [args.out_dir / f"sample_{cnt + i:04d}.npy" for i in range(batch_size)]
        todo = [not p.exists() and (cnt + i in args.indices) for (i, p) in enumerate(output_filenames)] # Whether the file should be generated
        if not any(todo):
            print(f"Nothing to do for the batches {cnt} - {cnt + batch_size - 1}.")
        else:
            batch = batch.to(args.device)
            recon = infer_video(mode=args.inference_mode, model=model, diffusion=diffusion,
                                batch=batch, max_T=args.max_T, obs_length=args.obs_length,
                                assert_fits_gpu=args.assert_fits_gpu)
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
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--out_dir", default=None, help="Output directory for the generated videos. If None, defaults to a directory at samples/<checkpoint_dir_name>/<checkpoint_name>_<checkpoint_step>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference_mode", required=True,
                        choices=["simple-autoreg", "exp-past", "multi-granuality", "independent"])
    # Inference arguments
    parser.add_argument("--max_T", type=int, default=10,
                        help="Maximum length of the sequence that fits in the GPU memory.")
    parser.add_argument("--obs_length", type=int, default=36,
                        help="Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest.")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="If not None, only generate videos for the specified indices.")
    parser.add_argument("--assert_fits_gpu", action="store_true",
                        help="Assert that the batches fit in the GPU memory. If not, the program will exit."
                             " If this argument is not specified, the program will drop observed enough frames to make sure it fits in the GPU memory.")
    parser.add_argument("--dryrun_gpu_memory", action="store_true",
                        help="Dry run the GPU memory. If this argument is specified, the program will run the GPU memory test to figure out the maximum batch size that fits in the GPU memory.")
    args = parser.parse_args()

    drange = [-1, 1] # Range of the generated samples' pixel values

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = Namespace(**data["config"])
    # Load the model
    model, diffusion = create_video_model_and_diffusion(
        **args_to_dict(model_args, video_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
            state_dict
        )
    model = model.to(args.device)
    model.eval()
    # Load the test set
    dataset = get_test_dataset("minerl")#(dataset_name=model_args.dataset) # TODO: fix
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"Dataset size = {len(dataset)}")
    if args.dryrun_gpu_memory:
        dryrun_gpu_memory(args, model, diffusion, dataloader)
    else:
        main(args, model, diffusion, dataloader)
