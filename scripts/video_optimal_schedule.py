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
import functools

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
from improved_diffusion.rng_util import RNG

sys.path.insert(1, str(Path(__file__).parent.resolve()))
from video_nll import run_bpd_evaluation


# A dictionary of default model configs for the parameters newly introduced.
# It enables backward compatibility
default_model_configs = {"enforce_position_invariance": False,
                         "cond_emb_type": "channel"}


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


@torch.no_grad()
def get_mse_linspace(latent_frame_indices, candidate_idx, model, diffusion, dataloader, device,
                     num_timesteps, inference_step):
    obs_incides = [[candidate_idx] for _ in range(dataloader.batch_size)]
    lat_indices = [latent_frame_indices for _ in range(dataloader.batch_size)]
    mse_all = []
    for cnt, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)
        t_seq = diffusion.num_timesteps - 1 - np.linspace(0, diffusion.num_timesteps, num_timesteps, endpoint=False, dtype=int)
        metrics = run_bpd_evaluation(
            model=model, diffusion=diffusion,
            batch=batch, clip_denoised=True,
            obs_indices=obs_incides[:len(batch)], lat_indices=lat_indices[:len(batch)],
            t_seq=t_seq)
        metrics = {k : v / num_timesteps * diffusion.num_timesteps for k, v in metrics.items()}
        mse = metrics["mse"]
        assert mse.ndim == 1
        mse_all.append(mse)
    mse_all = np.concatenate(mse_all, axis=0)
    return mse_all.mean(), mse_all.std()


@torch.no_grad()
def get_mse_random(latent_frame_indices, candidate_idx, model, diffusion, dataloader, device,
                   num_timesteps, inference_step):
    obs_incides = [[candidate_idx] for _ in range(dataloader.batch_size)]
    lat_indices = [latent_frame_indices for _ in range(dataloader.batch_size)]
    mse_all = []
    cnt = 0
    for batch, _ in dataloader:
        B = len(batch)
        batch = batch.to(device)
        t_seq = np.array([[np.random.RandomState(inference_step * 1000 + cnt + i).randint(diffusion.num_timesteps)] for i in range(B)])
        metrics = run_bpd_evaluation(
            model=model, diffusion=diffusion,
            batch=batch, clip_denoised=True,
            obs_indices=obs_incides[:len(batch)], lat_indices=lat_indices[:len(batch)],
            t_seq=t_seq)
        metrics = {k : v / t_seq.shape[1] * diffusion.num_timesteps for k, v in metrics.items()}
        mse = metrics["mse"]
        assert mse.ndim == 1
        mse_all.append(mse)
        cnt += B
    mse_all = np.concatenate(mse_all, axis=0)
    return mse_all.mean(), mse_all.std()


def force_nearby(latent_frame_indices, obs_frame_indices, done_frame_indices):
    done_frame_indices = sorted(list(done_frame_indices))
    # Closest frame before the latent frames
    idx = None
    for x in done_frame_indices:
        if x < min(latent_frame_indices) and x not in latent_frame_indices:
            idx = x
        else:
            break
    if idx is not None:
        obs_frame_indices.add(idx)
    # Closest frame after the latent frames
    idx = None
    for x in done_frame_indices[::-1]:
        if x > max(latent_frame_indices) and x not in latent_frame_indices:
            idx = x
        else:
            break
    if idx is not None:
        obs_frame_indices.add(idx)


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
            #continue
        n_to_condition_on = frame_indices_iterator._max_frames - len(latent_frame_indices)
        obs_frame_indices = set()
        if "force-nearby" in args.optimality:
            force_nearby(latent_frame_indices=latent_frame_indices,
                         obs_frame_indices=obs_frame_indices,
                         done_frame_indices=frame_indices_iterator._done_frames)
        if "linspace-t" in args.optimality:
            get_metric_fn = get_mse_linspace
        elif "random-t" in args.optimality:
            get_metric_fn = get_mse_random
        else:
            raise ValueError(f"Unrecognized optimality {args.optimality}.")
        while len(obs_frame_indices) < min(len(frame_indices_iterator._done_frames), n_to_condition_on):
            # Find the next best observed frame index
            best_idx = -1
            best_metric = np.inf
            for candidate_idx in frame_indices_iterator._done_frames:
                if candidate_idx in latent_frame_indices or candidate_idx in obs_frame_indices:
                    # Skip the latent frames (these are just added to the done list by the InferenceStrategyBase class)
                    # Also skip the frames that are already in the observed list
                    continue
                mean, std = get_metric_fn(latent_frame_indices=latent_frame_indices,
                                          candidate_idx=candidate_idx,
                                          model=model, diffusion=diffusion,
                                          dataloader=dataloader, device=args.device,
                                          num_timesteps=args.num_timesteps,
                                          inference_step=cnt)
                if mean < best_metric:
                    best_metric, best_idx = mean, candidate_idx
                if verbose:
                    print(f"Candidate frame {candidate_idx}: ({mean}, {std}) --- best so far: {obs_frame_indices} + {best_idx} ({best_metric})")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_dir", default=None, help="Path to the evaluation directory for the given checkpoint. If None, defaults to resutls/<checkpoint_dir_subset>/<checkpoint_name>.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Inference arguments
    parser.add_argument("--optimality", type=str, default="linspace-t",
        choices=["linspace-t", "random-t",
                 "linspace-t-force-nearby", "random-t-force-nearby"])
    parser.add_argument("--inference_mode", required=True, choices=inference_util.inference_strategies.keys())
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
    parser.add_argument("--subset_size", type=int, default=None,
                        help="If not None, only use a subset of the dataset. Default is 50.")
    parser.add_argument("--num_timesteps", type=int, default=None,
                        help="Number of timesteps to use for estimating the ELBO. Only used in mse-linsapce.")
    parser.add_argument("--step", type=int, default=None, help="Which step of inference to produce optimal observations for. Used for parallel sampling on multiple machines.")
    # Job submission arguments
    parser.add_argument("--submit", action="store_true", help="If given, figures out which steps of the optimal schedule are remaining to be optimized, then submits an array job doing them.")
    parser.add_argument("--max_slurm_array", type=int, default=None, help="If given, will use it as a limit on the number of concurrently running array jobs.")
    args = parser.parse_args()
    if args.num_timesteps is None:
        args.num_timesteps = 10
    if args.subset_size is None:
        args.subset_size = 10 if "linspace-t" in args.optimality else 100

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
    args.eval_dir = test_util.get_model_results_path(args) / test_util.get_eval_run_identifier(args)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.eval_dir / "optimal_schedule.pt"
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
