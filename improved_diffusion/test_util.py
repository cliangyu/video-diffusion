from filelock import FileLock
from pathlib import Path


class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)


def get_samples_root_path(args, model_step):
    """
        args is expected to have the following attributes:
        - use_ddim
        - timesptep_respacing
        - outdir
    """
    # Extract the diffusion sampling arguments string (DDIM/respacing)
    postfix = ""
    if args.use_ddim:
        postfix += "_ddim"
    if args.timestep_respacing != "":
        postfix += "_" + f"respace{args.timestep_respacing}"

    # Create the output directory (if does not exist)
    if args.out_dir is None:
        checkpoint_path = Path(args.checkpoint_path)
        name = f"{checkpoint_path.stem}_{model_step}"
        if postfix != "":
            name += postfix
        path = None
        for idx, x in enumerate(checkpoint_path.parts):
            if "checkpoint" in x:
                path = Path(*(checkpoint_path.parts[idx+1:-1]))
                break
        assert path is not None
        return Path("samples") / path.parent / name
    else:
        return Path(args.out_dir)


def get_eval_run_identifier(args):
    """
        args is expected to have the following attributes:
        - inference_mode
        - max_frames
        - step_size
        - T
        - obs_length
    """
    return f"{args.inference_mode}_{args.max_frames}_{args.step_size}_{args.T}_{args.obs_length}"