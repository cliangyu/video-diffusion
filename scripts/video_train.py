"""
Train a diffusion model on videos.
"""

import argparse
import os, sys
import wandb
from pathlib import Path


from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_video_data, default_T_dict, default_image_size_dict
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    video_model_and_diffusion_defaults,
    create_video_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


os.environ["MY_WANDB_DIR"] = "none"
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"
    if "WANDB_DIR_DRYRUN" in os.environ:
        os.environ["MY_WANDB_DIR"] = os.environ["WANDB_DIR_DRYRUN"]


def num_available_cores():
    # Copied from pytorch source code https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest or 1


def main():
    args = create_argparser().parse_args()
    if args.num_workers == -1:
        # Set the number of workers automatically.
        args.num_workers = max(num_available_cores() - 1, 1)
        print(f"num_workers is not specified. It is automatically set to \"number of cores - 1\" = {args.num_workers}")

    video_length = default_T_dict[args.dataset]
    default_T = video_length
    default_image_size = default_image_size_dict[args.dataset]
    args.T = default_T if args.T == -1 else args.T
    args.image_size = default_image_size
    if args.rp_alpha is None:
        assert args.rp_beta is None
        args.rp_alpha = args.rp_beta = args.rp_gamma = args.T
    assert args.rp_alpha is not None and args.rp_beta is not None and args.rp_gamma is not None
    args.rp_alpha = int(args.rp_alpha)
    args.rp_beta = int(args.rp_beta)
    args.rp_gamma = int(args.rp_gamma)
    assert args.rp_beta >= args.rp_alpha
    
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(
        config=args,
        resume=bool(args.resume_id),
        id=args.resume_id if args.resume_id else None
    )
    logger.log("creating video model and diffusion...")
    model, diffusion = create_video_model_and_diffusion(
        **args_to_dict(args, video_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_video_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        T=args.T,
        num_workers=args.num_workers,
        data_path=args.data_path,
    )

    logger.log("training...")
    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        do_inefficient_marg=args.do_inefficient_marg,
        n_valid_batches=args.n_valid_batches,
        n_valid_repeats=args.n_valid_repeats,
        n_interesting_masks=args.n_interesting_masks,
        max_frames=args.max_frames,
        T=args.T,
        mask_distribution=args.mask_distribution,
        pad_with_random_frames=args.pad_with_random_frames,
        observed_frames=args.observed_frames,
        args=args,
    )
    if args.just_visualise:
        train_loop.visualise()
        exit()
    if args.just_save_masks > 0:
        train_loop.save_masks(args.just_save_masks)
        exit()
    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        dataset="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        sample_interval=50000,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        do_inefficient_marg=False,
        n_valid_batches=1,
        n_valid_repeats=2,
        valid_microbatch=-1,
        n_interesting_masks=3,
        max_frames=10,
        save_latest_only=False,  # If False, keeps all the checkpoints saved during training.
        resume_id="",
        mask_distribution="differently-spaced-groups",   # can also do "consecutive-groups" or "autoregressive-{i}", or "differently-spaced-groups-no-marg"
        just_visualise=False,
        just_save_masks=0,
        num_workers=-1,     # Number of workers to use for training dataloader. If not specified, uses the number of available cores on the machine.
        pad_with_random_frames=True,
        fake_seed=1,  # the random seed is never set, but this lets us run sweeps with is as if it controls the seed
        observed_frames='x_t_minus_1',  # the input of observed frames, case 1: 'x_0', case 2: 'x_t', case 3: 'x_t_minus_1'
        data_path=None,  # assign data path
    )
    defaults.update(video_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
