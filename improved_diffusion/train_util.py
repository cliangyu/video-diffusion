import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb
from time import time

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .rng_util import rng_decorator

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        sample_interval=None,
        do_inefficient_marg=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.do_inefficient_marg = do_inefficient_marg
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.valid_batch = next(self.data)[0][:self.microbatch]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def sample_video_mask(self, data, mask_type, exclude=None):
        like = data[..., :1, :1, :1]  # B x T x 1 x 1 x 1
        if exclude is None:
            exclude = th.zeros_like(like)
        B, T, *_ = like.shape
        if mask_type == 'zero':
            return th.zeros_like(like)
        elif mask_type == 'obs':
            mask = th.zeros_like(like)
            for r, row in enumerate(mask):
                n_obs = np.random.randint(T//5, size=())
                pos = np.random.randint(T-n_obs+1, size=())
                row[pos:pos+n_obs] = 1.
        elif mask_type == 'marg':
            mask = th.zeros_like(like)
            np_exclude = exclude.view(B, T).int().cpu().numpy()
            for r, row in enumerate(mask):
                # sample such that up to 10 things are excluded or marginalised
                max_unmarg_unexcluded = 10
                n_excluded = np_exclude[r].sum()
                max_unmarg = max_unmarg_unexcluded - n_excluded
                n_unmarg = np.random.randint(low=1, high=max_unmarg, size=())
                unmarg_indices = np.random.choice(T-n_excluded, size=n_unmarg, replace=False)
                unmarg_indices = th.nonzero(1-exclude[r], as_tuple=True)[0][unmarg_indices]
                row[exclude[r]==0] = 1.
                row[unmarg_indices] = 0.
        if exclude is not None:
            mask = mask * (1 - exclude)
        return mask

    def sample_all_masks(self, batch):
        obs_mask = self.sample_video_mask(batch, 'obs')
        pt, ft = ('marg', 'zero') if self.do_inefficient_marg else ('zero', 'marg')
        latent_mask = 1 - obs_mask
        partly_marg_mask = self.sample_video_mask(batch, pt, exclude=1-latent_mask)
        latent_mask = latent_mask * (1-partly_marg_mask)
        fully_marg_mask = self.sample_video_mask(batch, ft, exclude=1-latent_mask)
        dynamics_mask = latent_mask * (1-fully_marg_mask)
        # delete as many frames as possible fiven fully_marg_mask
        not_fully_marg_mask, (batch, obs_mask, partly_marg_mask, dynamics_mask), frame_indices =\
            self.gather_unmasked_elements(
                (1-fully_marg_mask), [batch, obs_mask, partly_marg_mask, dynamics_mask]
        )
        fully_marg_mask = 1 - not_fully_marg_mask
        return batch, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, dynamics_mask

    def gather_unmasked_elements(self, mask, tensors):
        B, T, *_ = mask.shape
        mask = mask.view(B, T)  # remove unit C, H, W dims
        effective_T = mask.sum(dim=1).max().int()
        new_mask = th.zeros_like(mask[:, :effective_T])
        indices = th.zeros_like(mask[:, :effective_T], dtype=th.int64)
        new_tensors = [th.zeros_like(t[:, :effective_T]) for t in tensors]
        for b in range(B):
            instance_T = mask[b].sum().int()
            new_mask[b, :instance_T] = 1
            indices[b, :instance_T] = mask[b].nonzero().flatten()
            for new_t, t in zip(new_tensors, tensors):
                new_t[b, :instance_T] = t[b][mask[b]==1]
        return new_mask.view(B, effective_T, 1, 1, 1), new_tensors, indices

    def run_loop(self):
        last_sample_time = time()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            batch, cond = next(self.data)

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.sample_interval is not None and self.step != 0 and self.step % self.sample_interval == 0:
                self.log_samples()
                logger.logkv('time_between_samples', time()-last_sample_time)
                last_sample_time = time()
            self.step += 1
            if self.step == 1:
                gather_and_log_videos('data', batch)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, dynamics_mask = self.sample_all_masks(micro)
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={**micro_cond, 'frame_indices': frame_indices, 'obs_mask': obs_mask,
                              'partly_marg_mask': partly_marg_mask,
                              'fully_marg_mask': fully_marg_mask, 'x0': micro},
                dynamics_mask=dynamics_mask,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    @rng_decorator(seed=0)
    def log_samples(self):
        sample_start = time()
        self.model.eval()
        logger.log("sampling...")
        orig_batch = self.valid_batch.to(dist_util.dev())
        batch, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, dynamics_mask = \
            self.sample_all_masks(orig_batch)

        img_size = batch.shape[-1]
        # copied from scripts/image_sample.py ---------------------------------
        sample_fn = (
            self.diffusion.p_sample_loop
        )
        sample = sample_fn(
            self.model,
            batch.shape,
            clip_denoised=True,
            model_kwargs={
                'frame_indices': frame_indices,
                'x0': batch, 'obs_mask': obs_mask,
                'partly_marg_mask': partly_marg_mask,
                'fully_marg_mask': fully_marg_mask},
            dynamics_mask=dynamics_mask,
        )
        # ---------------------------------------------------------------------
        batch_vis = th.zeros_like(orig_batch)
        batch_is_latent = dynamics_mask.view(sample.shape[:2]).bool()
        batch_is_obs = obs_mask.view(sample.shape[:2]).bool()
        orig_batch[:, :, :1] = 0   # mutilate observed frames
        for vis, is_latent, is_obs, frame_indices_element, data_element, sampled_element in zip(
                batch_vis, batch_is_latent, batch_is_obs, frame_indices, orig_batch, sample
        ):
            obs_indices = frame_indices_element[is_obs]
            vis[obs_indices] = data_element[obs_indices]
            latent_indices = frame_indices_element[is_latent]
            vis[latent_indices] = sampled_element[is_latent]
        gather_and_log_videos('sample', batch_vis)
        logger.log("sampling complete")
        logger.logkv('sampling_time', time()-sample_start)
        self.model.train()


def gather_and_log_videos(name, array):
    """
    Unnormalises and logs videos given as B x T x C x H x W tensors.
    """
    array = array.cuda()
    array = ((array + 1) * 127.5).clamp(0, 255).to(th.uint8)
    array = array.permute(0, 1, 3, 4, 2)
    array = array.contiguous()
    gathered_arrays = [th.zeros_like(array) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_arrays, array)  # gather not supported with NCCL
    images = np.concatenate(
        [array.cpu().numpy() for array in gathered_arrays],
        axis=0
    )
    divider = images[:, 0, :, :1]*0 + 127
    T = array.shape[1]
    images = np.concatenate(
        [np.concatenate([images[:, t], divider], axis=2) for t in range(T)],
        axis=2
    )
    dist.barrier()
    logger.logkvs({f'{name}-{i}': wandb.Image(image) for i, image in enumerate(images)})


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
