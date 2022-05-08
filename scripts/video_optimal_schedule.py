from ast import parse
import sched
from shutil import move
import torch
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
import os, sys
import subprocess
from tqdm.auto import tqdm
from pathlib import Path
import json

from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion import dist_util
from improved_diffusion.image_datasets import get_train_dataset
from improved_diffusion.script_util import str2bool
from improved_diffusion import inference_util
from improved_diffusion import test_util


# A dictionary of default model configs for the parameters newly introduced.
# It enables backward compatibility
default_model_configs = {"enforce_position_invariance": False,
                         "cond_emb_type": "channel"}


def run_bpd_evaluation(model, diffusion, batch, clip_denoised, obs_indices, lat_indices, t_seq=None):
    x0 = torch.zeros_like(batch[:, :len(obs_indices[0]) + len(lat_indices[0])])
    obs_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    lat_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    kinda_marg_mask = torch.zeros_like(x0[:, :, :1, :1, :1])
    frame_indices = torch.zeros_like(x0[:, :, 0, 0, 0]).int()
    for i, (obs_i, lat_i) in enumerate(zip(obs_indices, lat_indices)):
        x0[i, :len(obs_i)] = batch[i, obs_i]
        obs_mask[i, :len(obs_i)] = 1.
        frame_indices[i, :len(obs_i)] = torch.tensor(obs_i)
        x0[i, len(obs_i):len(obs_i)+len(lat_i)] = batch[i, lat_i]
        lat_mask[i, len(obs_i):len(obs_i)+len(lat_i)] = 1.
        frame_indices[i, len(obs_i):len(obs_i)+len(lat_i)] = torch.tensor(lat_i)
    model_kwargs = dict(frame_indices=frame_indices,
                        x0=x0,
                        obs_mask=obs_mask,
                        latent_mask=lat_mask,
                        kinda_marg_mask=kinda_marg_mask)
    metrics = diffusion.calc_bpd_loop_subsampled(
        model, x0, clip_denoised=clip_denoised, model_kwargs=model_kwargs, t_seq=t_seq, latent_mask=lat_mask
    )
    n_latents_batch = lat_mask.flatten(start_dim=1).sum(dim=1) # Number of latent frames in each video. Shape: (B,)

    metrics = {k: v.sum(dim=1) if v.ndim > 1 else v for k, v in metrics.items()}
    # sum (rather than mean) over frame dimension by multiplying by number of frames
    metrics = {k: v*n_latents_batch for (k, v) in metrics.items()}
    metrics = {k: v.detach().cpu().numpy() for k, v in metrics.items()}
    return metrics


@torch.no_grad()
def get_mse(latent_frame_indices, candidate_idx, model, diffusion, dataloader, device):
    obs_incides = [[candidate_idx] for _ in range(dataloader.batch_size)]
    lat_indices = [latent_frame_indices for _ in range(dataloader.batch_size)]
    mse_all = []
    for cnt, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)
        t_seq = diffusion.num_timesteps - 1 - np.linspace(0, diffusion.num_timesteps, 10, endpoint=False, dtype=int)
        #np.random.RandomState(123 + cnt).randint(0, diffusion.num_timesteps, size=len(batch))
        metrics = run_bpd_evaluation(
            model=model, diffusion=diffusion,
            batch=batch, clip_denoised=True,
            obs_indices=obs_incides[:len(batch)], lat_indices=lat_indices[:len(batch)],
            t_seq=t_seq)
        mse = metrics["mse"]
        assert mse.ndim == 1
        mse_all.append(mse)
    mse_all = np.concatenate(mse_all, axis=0)
    return mse_all.mean(), mse_all.std()


def main(args, model, diffusion, dataloader, schedule_path, verbose=True):
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None
    frame_indices_iterator = inference_util.inference_strategies[args.inference_mode](
        video_length=args.T, num_obs=args.obs_length, max_frames=args.max_frames, step_size=args.step_size
    )
    inference_schedule = {} # A dictionary from "inference step" to the list of optimal observed frame indices.
    schedule_path = Path(schedule_path)
    if schedule_path.exists():
        with test_util.Protect(schedule_path):
            saved_schedule = torch.load(schedule_path)
    else:
        saved_schedule = {}
    for cnt, (_, latent_frame_indices) in enumerate(tqdm(frame_indices_iterator)):
        if task_id is not None and cnt != task_id:
            print(f"Skipping inference step {cnt}; not for this instance of array job.")
            continue
        if cnt in saved_schedule:
            print(f"Skipping inference step {cnt}; already done.")
            continue
        n_to_condition_on = frame_indices_iterator._max_frames - len(latent_frame_indices)
        obs_frame_indices = set()
        while len(obs_frame_indices) < min(len(frame_indices_iterator._done_frames), n_to_condition_on):
            # Find the next best observed frame index
            best_idx = -1
            best_mse = np.inf
            for candidate_idx in frame_indices_iterator._done_frames:
                if candidate_idx in latent_frame_indices or candidate_idx in obs_frame_indices:
                    # Skip the latent frames (these are just added to the done list by the InferenceStrategyBase class)
                    # Also skip the frames that are already in the observed list
                    continue
                mean, std = get_mse(latent_frame_indices=latent_frame_indices,
                                    candidate_idx=candidate_idx,
                                    model=model, diffusion=diffusion,
                                    dataloader=dataloader, device=args.device)
                if mean < best_mse:
                    best_mse, best_idx = mean, candidate_idx
                if verbose:
                    print(f"Candidate frame {candidate_idx}: ({mean}, {std}) --- best so far: {obs_frame_indices} + {best_idx} ({best_mse})")
            obs_frame_indices.add(best_idx)
        obs_frame_indices = sorted(list(obs_frame_indices))
        inference_schedule[cnt] = obs_frame_indices
        print(f"Step #{cnt}:\n\tLatent: {latent_frame_indices}\n\tObserved: {obs_frame_indices}")
        with test_util.Protect(schedule_path):
            # Re-load the test schedule, in case it was modified by another process.
            saved_schedule = torch.load(schedule_path) if schedule_path.exists() else {}
            for k,v in inference_schedule.items():
                assert k not in saved_schedule, f"Found {k} in the saved schedule!"
                saved_schedule[k] = v
            torch.save(saved_schedule, schedule_path)
            inference_schedule = {}


def submit(remaining_steps, time="3:00:00", max_slurm_array=None):
    if len(remaining_steps) == 0:
        print("Nothing left to do!")
        return
    SUBMISSION_CMD = "~/.dotfiles/job_submission/submit_job.py"
    SUBMISSION_ARGS = f"--mem=32G --gres=gpu:1 --time {time} --mail-type END,FAIL"
    array_arg = "--array " + ",".join(map(str, remaining_steps)) + ("" if max_slurm_array is None else f"%{max_slurm_array}")
    SUBMISSION_ARGS = f"{SUBMISSION_ARGS} {array_arg}"
    job_name = "viddiff-opt-sched"

    submission_args = f"-J {job_name} {SUBMISSION_ARGS}"
    # Exctract script arguments, drop out --submit
    script_args = " ".join([arg for arg in sys.argv if arg != "--submit"])
    # Construct the full job submission command
    cmd = " ".join([SUBMISSION_CMD, submission_args, "--", "python", script_args])
    print("--> Submitting a job with the following command:\n> ", cmd)
    print("#######################################################\n")
    key_in = None
    while key_in not in ["y", "n", ""]:
        key_in = input("Proceed? (y/N)").lower()
        print(key_in)
    if key_in == "":
        key_in = "n"
    if key_in == "y":
        subprocess.call(cmd, shell=True)


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
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--T", type=int, default=None,
                        help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--subset_size", type=int, default=8,
                        help="If not None, only use a subset of the dataset. Default is 50.")
    parser.add_argument("--step", type=int, default=None, help="Which step of inference to produce optimal observations for. Used for parallel sampling on multiple machines.")
    # Job submission arguments
    parser.add_argument("--submit", action="store_true", help="If given, figures out which steps of the optimal schedule are remaining to be optimized, then submits an array job doing them.")
    parser.add_argument("--max_slurm_array", type=int, default=None, help="If given, will use it as a limit on the number of concurrently running array jobs.")
    args = parser.parse_args()

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
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

    args.optimal = True # Required for get_eval_run_identifier to work correctly.
    args.out_dir = test_util.get_model_results_path(args) / test_util.get_eval_run_identifier(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.out_dir / "optimal_schedule.pt"
    print(f"Saving the optimal inference schedule to {schedule_path}")

    if args.submit:
        # Figure out which steps are remaining
        saved_schedule = {} if not schedule_path.exists() else torch.load(schedule_path)
        frame_indices_iterator = inference_util.inference_strategies[args.inference_mode](
            video_length=args.T, num_obs=args.obs_length, max_frames=args.max_frames, step_size=args.step_size)
        num_steps = len(list(frame_indices_iterator))
        remaining_steps = [step for step in range(num_steps) if step not in saved_schedule]
        submit(remaining_steps, time="3:00:00", max_slurm_array=args.max_slurm_array)
        quit()

    # Load the test set
    dataset = get_train_dataset(dataset_name=model_args.dataset)
    print(f"Dataset size = {len(dataset)}")
    # Prepare the indices
    if args.subset_size is not None:
        indices = np.random.RandomState(123).choice(len(dataset), args.subset_size, replace=False)
        print(f"Only generating predictions a randomly chosen subset of size {args.subset_size} videos of the dataset.")
        dataset = torch.utils.data.Subset(dataset, indices)
    print(f"Dataset size (after subsampling) = {len(dataset)}")
    # Prepare the dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Generate the samples
    main(args, model, diffusion, dataloader, schedule_path=schedule_path)
