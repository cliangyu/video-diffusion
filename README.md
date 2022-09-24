# Video diffusion

## Prepare Dataset

```bash
### UCF101 Dataset
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
### BAIR Pushing Dataset
wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
### MineRL Dataset
pip install tensorflow
pip install tensorflow_datasets
conda update ffmpeg
python datasets/minerl.py
### Maze Dataset
python datasets/maze.py
```

## Directory structures

### Checkpoints

Checkpoints have the following directory structure:

```bash
checkpoints
├── .../<wandb_id>
│   ├── model_latest.pt
│   ├── ema_<ema_rate>_latest.pt
│   ├── opt_latest.pt
│   ├── model_<step>.pt
│   ├── ema_<ema_rate>_<step>.pt
│   └── opt_<step>.pt
└── ... (other wnadb runs)
```

### Results

Results have the following directory structure:

```bash
results
├── .../<wandb_id>
│   ├── <checkpoint_name>
│   │   ├── <inference_mode_str>        (we call this directory the "evaluation directory")
│   │   │   ├── model_config.json       (includes the training arguments of the checkpoint)
│   │   │   ├── videos
│   │   │   │  ├── <name-1>.gif
│   │   │   │  ├── <name-2>.gif
│   │   │   │  ├── ...
│   │   │   │  └── <name-n>.gif
│   │   │   ├── samples
│   │   │   │  ├── <name-1>.npy
│   │   │   │  ├── <name-2>.npy
│   │   │   │  ├── ...
│   │   │   │  └── <name-n>.npy
│   │   │   ├── elbos
│   │   │   │  ├── <name-1>.npy
│   │   │   │  ├── <name-2>.npy
│   │   │   │  ├── ...
│   │   │   │  └── <name-n>.npy
│   │   │   └── <metrics_name>.pkl
│   │   └── ... (other inference modes)
│   └── ... (other checkpoints of the same run)
└── ... (other wnadb runs)
```

In this directory structure,

- `checkpoint_name` is the name of the checkpoint file, appended with `_<checkpoint_step>` if the checkpoint name ends with `_latest`.
- `<inference_mode_str>` is a string defining the inference mode, including its various parameters. At the time of writing, it is `<inference_mode>_<max_frames>_<step_size>_<T>_<obs_length>` where `<inference_mode>` is one of the inference
  strategies, with `_optimal` appended if using the optimal observations.

## Inference

The script `scripts/video_sample.py` is used to sample a video from the model.

### Basic usage

An example usage of the script is:

```bash
python scripts/video_sample.py <path-to-checkpoint> --inference_mode independent --step_size 7
```

It has the following arguments:

- `checkpoint_path` (required): A required positional argument identifying path to the checkpoint file. It is shown by `<path-to-checkpoint>` in the example above.
- `--inference_mode` (required): Inference mode. Currently, we supprt `autoreg`, `independent`, and `exp-past`. You can easily add your own mode as described later.
- `--step_size` (optional): Number of frames to predict in each prediciton step. Default is 1.
- `--obs_length` (optional): Number of observed frames. It will observe this many frames from the beginning of the video and predict the rest. Default is 36.
- `--max_frames` (optional): Maximum number of video frames (observed or latent) allowed to pass to the model at once. Default is what the model was trained with.
- `--num_samples` (optional): Number of samples to generate for each test video. Default is 1.
- `--T` (optional): Length of the videos. If not specified, it will be inferred from the dataset.
- `--subset_size` (optional): If given, only uses a (random) subset of the test set with the size specified. Defaults to the whole test set.
- `--batch_size` (optional): Batch size. Default is 8.
- `--eval_dir` (optional): Output directory for the evaluations aka "evaluation directory". Default is `resutls/<checkpoint_dir_subset>/<checkpoint_name>`. Here, `<checkpoint_dir_subset>` is a subset of the checkpoint path
  after `.*checkpoints.*/`, and `<checkpoint_name>` is the checkpoint's `.pt` file name, appended with the checkpoint step if it ends with `_latest`.

Running the script will generate `.npy` files for videos in the test set. Each generated video is saved at `<out_dir>/samples/sample_<video-idx>-<sample-idx>.npy` where `<video-idx>` is the test set index to the video and `<sample-idx>`
enumerates the sample generated for that video.

### Advanced usage

There are a few more arguments that can be used to further tune the sampling process.

- `--use_ddim` (optional): If set to `True`, the model will use the DDIM for sampling. Default is `False`.
- `--timestep_respacing` (optional): Specifies the number of diffusion steps for video generation. Supports integers or `ddim<int>`. Default is the number of diffusion steps the model was trained with.
- `--indices` (optional): If given, only generates videos for the specified list of indices. Used for handling parallelized generation.
- `--sample_idx` (optional): Sampled images will have this specific index. Used for parallel sampling on multiple machines. If this argument is given, --num_samples is ignored.

If `SLURM_ARRAY_TASK_ID` is defined in environment variables, it only generates completions for one batch of test videos. The value of `SLURM_ARRAY_TASK_ID` identifies the index to the batch.

An example command to generate samples on multiple machines in parallel is:

`` RUN_ID=11n992cv; STEP=7; for S_IDX in `seq 0 9`; submit_job -J viddiff-sample --mem 16G --gres=gpu:1 --time 3:00:00 --array 0-29 -- python scripts/video_sample.py checkpoints/${RUN_ID}/ema_0.9999_latest.pt --batch_size 8 --inference_mode autoreg --step_size $STEP --T 300 --sample_idx $S_IDX; done ``

### Adding new inference modes

Inference modes are defined in `improved_diffusion/inference_utils.py`. It is also where new inference modes can be easily added. Each inference mode is defined by a class that inherits from `InferenceStrategyBase`. It maintains attributes
storing video length (`self._video_length`), max_T (`self._max_T`), frame idices that are generated by the model (`self._done_frames`), obseved frame indices (`self._obs_frames`), and step size (`self._step_size`).

An inference mode class should override the `next_indices` method. This argument-less method returns a tuple of `(obs_frame_indices, latent_frame_indices)` mainly based on attributes maintained by the parent class (
particularly, `self._done_frames`, `self._done_frames`, `self._max_T` and `self._step_size`). The parent class takes care of updating these attributes and also defining an iterable object for the downstream functions.

Inference mode classes can also optionally override the `is_done` class which identifies if the inference process is done (i.e., if the model has generated all the latent frames). By default, it is assumed that the inference process is done
when the number of done frames is equal to the number of latent frames (video length - observed frames). However, for some inference modes like Gibbs-like sampling, one might want to continue refining the frames even after they are generted
by the model.

For example, here is the implementation of autoregressive inference mode:

```python
class Autoregressive(InferenceStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        obs_frame_indices = sorted(self._done_frames)[-(self._max_T - self._step_size):]
        first_idx = obs_frame_indices[-1] + 1
        latent_frame_indices = list(range(first_idx, min(first_idx + self._step_size, self._video_length)))
        return obs_frame_indices, latent_frame_indices
```

## Evaluation

The script `scripts/video_eval.py` and `scripts/video_eval_fvd.py` are used for evaluating the model by different metrics (at the time of writing, PSNR, SSIM, LPIPS and FVD are supported) once samples from the model are generated.

Example usage:

```
python scripts/video_eval.py --eval_dir results/second-batch-400k-iters/3kdr4q5k/ema_0.9999_400000/hierarchy-2_optimal_20_10_300_36/ --num_samples 3
```

It will create a file at `<eval_dir>/<metrics_name>.pkl` containting a dicrionary from metric names to metric values. At the time of writing, `<metrics_name>` is `metrics_<number_of_test_videos_considered>-<number_of_samples_per_video>-<T>`

For the list and description of all arguments run `python scripts/video_eval.py --help` or `python scripts/video_eval_fvd.py --help`.
        