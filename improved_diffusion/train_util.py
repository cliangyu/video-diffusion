import copy
import functools
import os
from pathlib import Path

import shutil
import glob
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb
from time import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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
from .rng_util import RNG, rng_decorator

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
            T,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            sample_interval=None,
            do_inefficient_marg=True,
            n_valid_batches=1,
            n_valid_repeats=1,
            max_frames=10,
            n_interesting_masks=3,
            mask_distribution="differently-spaced-groups",
            pad_with_random_frames=True,
            observed_frames='x_t_minus_1',
            args=None
    ):
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        print('\n\n RUNNING INIT TRAIN LOOP WITH RANK', current_rank, '\n\n')
        assert args is not None
        self._args = args  # This is only to be used when saving the model
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.valid_microbatch = args.valid_microbatch if args.valid_microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.do_inefficient_marg = do_inefficient_marg
        self.T = T
        self.max_frames = max_frames
        self.log_interval = log_interval
        self.sample_interval = sample_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint if resume_checkpoint else find_resume_checkpoint(self._args)
        print(self.resume_checkpoint)
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_batch = self.batch_size * world_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self.observed_frames = observed_frames

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, 250000)
        if self.resume_checkpoint:
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
            if dist.is_initialized() and dist.get_world_size() > 1:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                self.ddp_model = self.model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        self.n_valid_batches = n_valid_batches
        self.n_valid_repeats = n_valid_repeats
        self.n_interesting_masks = n_interesting_masks
        self.mask_distribution = mask_distribution
        self.pad_with_random_frames = pad_with_random_frames
        with RNG(0):
            self.valid_batches = [next(self.data)[0][:self.valid_microbatch]
                                  for i in range(self.n_valid_batches)]
        if dist.is_initialized() and dist.get_rank() == 0:
            wandb.log({"num_parameters": sum(p.numel() for p in model.parameters())})

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
                self.step = state_dict["step"]
                self.model.load_state_dict(
                    state_dict["state_dict"]
                )
            else:
                # TODO: there might some codes to load model for single GPU
                pass
        if dist.is_initialized():
            dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.step, rate,
                                             save_latest_only=self._args.save_latest_only)
        if ema_checkpoint:
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict['state_dict'])
            else:
                # TODO: there might some codes to load model for single GPU
                pass
        else:
            print(f"Failed to find EMA checkpoint for rate {rate} and main checkpoint {main_checkpoint}.")
            raise Exception
        if dist.is_initialized():
            dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        filename = "opt_latest.pt" if self._args.save_latest_only else f"opt_{(self.step):06d}.pt"
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), filename
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print(f"Failed to find optimizer checkpoint {opt_checkpoint}.")
            raise Exception

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def sample_some_indices(self, max_indices, T):
        s = th.randint(low=1, high=max_indices + 1, size=())
        max_scale = T / (s - 0.999)
        if self.mask_distribution in ["one-group", "differently-spaced-groups", "differently-spaced-groups-no-marg"] or 'linspace' in self.mask_distribution:
            scale = np.exp(np.random.rand() * np.log(max_scale))
        elif self.mask_distribution == "consecutive-groups":
            scale = 1
        else:
            raise NotImplementedError
        pos = th.rand(()) * (T - scale * (s - 1))
        indices = [int(pos + i * scale) for i in range(s)]
        # do some recursion if we have somehow failed to satisfy the consrtaints
        if all(i < T and i >= 0 for i in indices):
            return indices
        else:
            print('warning: sampled indices', [int(pos + i * scale) for i in range(s)], 'trying again')
            return self.sample_some_indices(max_indices, T)

    def sample_all_masks(self, batch1, batch2=None, gather=True, set_masks={'obs': (), 'latent': (), 'kinda_marg': ()}):
        p_observed_latent_marg = th.tensor([0.33, 0.33, 0.33] if self.do_inefficient_marg else [0.5, 0.5, 0])
        N = self.max_frames
        B, T, *_ = batch1.shape
        masks = {k: th.zeros_like(batch1[:, :, :1, :1, :1]) for k in ['obs', 'latent', 'kinda_marg']}
        for obs_row, latent_row, marg_row in zip(*[masks[k] for k in ['obs', 'latent', 'kinda_marg']]):  # for each video
            if 'autoregressive' in self.mask_distribution:
                n_obs = int(self.mask_distribution.split('-')[1])
                n_latent = self.max_frames - n_obs
                start_i = th.randint(low=0, high=T - self.max_frames + 1, size=())
                obs_row[start_i:start_i + n_obs] = 1.
                latent_row[start_i + n_obs:start_i + n_obs + n_latent] = 1.
            elif 'linspace-no-obs' in self.mask_distribution:
                low, high, n = map(int, self.mask_distribution.split('-')[-3:])  # for frameskip set T=frameskip*(max_frames-1)+1
                indices = th.linspace(low, high, n).long()
                latent_row[indices] = 1.
            elif 'linspace' in self.mask_distribution:
                low, high, n = map(int, self.mask_distribution.split('-')[1:])  # for frameskip set T=frameskip*(max_frames-1)+1
                indices = th.linspace(low, high, n).long()
                latent_row[indices] = 1.
                while th.rand(size=()) > 0.5 and N - sum(obs_row) > 1:
                    index_indices = th.tensor(self.sample_some_indices(max_indices=N - sum(obs_row).int().item() - 1, T=N)).long()
                    obs_row[indices[index_indices]] = 1.
                    latent_row[indices[index_indices]] = 0.
            elif self.mask_distribution == 'uniform':
                n_frames = np.random.randint(1, self.max_frames, size=())
                n_obs = np.random.randint(0, n_frames, size=())
                indices = np.random.choice(T, size=n_frames.item(), replace=False)
                obs_row[indices[:n_obs]] = 1.
                latent_row[indices[n_obs:]] = 1.
            elif self.mask_distribution == 'uniform-no-marg':
                n_frames = self.max_frames
                n_obs = np.random.randint(0, n_frames, size=())
                indices = np.random.choice(T, size=n_frames, replace=False)
                obs_row[indices[:n_obs]] = 1.
                latent_row[indices[n_obs:]] = 1.
            elif self.mask_distribution == "differently-spaced-groups-no-marg":
                assert self.max_frames == T
                while th.rand(size=()) > 0.5 and N - sum(obs_row) > 1:
                    indices = th.tensor(self.sample_some_indices(max_indices=N - sum(obs_row).int().item() - 1, T=T))
                    obs_row[indices] = 1.
                latent_row += 1 - obs_row
            elif self.mask_distribution == "one-group":
                indices = self.sample_some_indices(max_indices=N, T=T)
                n_obs = np.random.randint(0, len(indices), size=())
                obs_indices = np.random.choice(indices, size=n_obs)
                obs_row[obs_indices] = 1.
                latent_indices = np.setdiff1d(indices, obs_indices)
                latent_row[latent_indices] = 1.
            elif 'groups' in self.mask_distribution:
                latent_row[self.sample_some_indices(max_indices=N, T=T)] = 1.
                while True:
                    mask_i = th.distributions.Categorical(probs=p_observed_latent_marg).sample()
                    mask = [obs_row, latent_row, marg_row][mask_i]
                    indices = th.tensor(self.sample_some_indices(max_indices=N, T=T))
                    taken = (obs_row[indices] + latent_row[indices] + marg_row[indices]).view(-1)
                    indices = indices[taken == 0]  # remove indices that are already used in a mask
                    if len(indices) > N - sum(obs_row) - sum(latent_row) - sum(marg_row):
                        break
                    mask[indices] = 1.
            else:
                raise NotImplementedError
        if len(set_masks['obs']) > 0:
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        represented_mask = (masks['obs'] + masks['latent'] + masks['kinda_marg']).clip(max=1)
        if not gather:
            return batch1, masks['obs'], masks['latent'], masks['kinda_marg']
        represented_mask, batch, (obs_mask, latent_mask, kinda_marg_mask), frame_indices = \
            self.gather_unmasked_elements(
                represented_mask, batch1, batch2, (masks['obs'], masks['latent'], masks['kinda_marg'])
            )
        return batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask

    def gather_unmasked_elements(self, mask, batch1, batch2, tensors):
        B, T, *_ = mask.shape
        mask = mask.view(B, T)  # remove unit C, H, W dims
        effective_T = self.max_frames if self.pad_with_random_frames else mask.sum(dim=1).max().int()
        new_mask = th.zeros_like(mask[:, :effective_T])
        indices = th.zeros_like(mask[:, :effective_T], dtype=th.int64)
        new_batch = th.zeros_like(batch1[:, :effective_T])
        new_tensors = [th.zeros_like(t[:, :effective_T]) for t in tensors]
        for b in range(B):
            instance_T = mask[b].sum().int()
            new_mask[b, :instance_T] = 1
            indices[b, :instance_T] = mask[b].nonzero().flatten()
            # select random frames in case we are doing padding with single frames
            indices[b, instance_T:] = th.randint_like(indices[b, instance_T:], high=T) if self.pad_with_random_frames else 0
            new_batch[b, :instance_T] = batch1[b][mask[b] == 1]
            new_batch[b, instance_T:] = (batch1 if batch2 is None else batch2)[b][indices[b, instance_T:]]
            for new_t, t in zip(new_tensors, tensors):
                new_t[b, :instance_T] = t[b][mask[b] == 1]
                new_t[b, instance_T:] = t[b][indices[b, instance_T:]]
        return new_mask.view(B, effective_T, 1, 1, 1), new_batch, new_tensors, indices

    def run_loop(self):
        if 'carla' not in self._args.dataset:
            gather_and_log_videos('data/', next(self.data)[0], log_as='both')
        last_sample_time = time()
        while (
                not self.lr_anneal_steps
                or self.step < self.lr_anneal_steps
        ):

            t_0 = time()
            self.run_step()
            logger.logkv("timing/step_time", time() - t_0)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.sample_interval is not None and self.step != 0 and (self.step % self.sample_interval == 0 or self.step == 5):
                self.log_samples()
                logger.logkv('timing/time_between_samples', time() - last_sample_time)
                last_sample_time = time()
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self):
        self.forward_backward()
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self):
        zero_grad(self.model_params)
        batch1 = next(self.data)[0]
        batch2 = next(self.data)[0] if self.pad_with_random_frames else None
        for i in range(0, batch1.shape[0], self.microbatch):
            micro1 = batch1[i:i + self.microbatch]
            micro2 = batch2[i:i + self.microbatch] if batch2 is not None else None
            micro, frame_indices, obs_mask, latent_mask, kinda_marg_mask = self.sample_all_masks(micro1, micro2)
            micro = micro.to(dist_util.dev())
            frame_indices = frame_indices.to(dist_util.dev())
            obs_mask = obs_mask.to(dist_util.dev())
            latent_mask = latent_mask.to(dist_util.dev())
            kinda_marg_mask = kinda_marg_mask.to(dist_util.dev())

            last_batch = (i + self.microbatch) >= batch1.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            loss_mask = (1 - obs_mask - kinda_marg_mask) if self.pad_with_random_frames else latent_mask
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={'frame_indices': frame_indices, 'obs_mask': obs_mask,
                              'latent_mask': latent_mask, 'kinda_marg_mask': kinda_marg_mask,
                              'x0': micro, 'observed_frames': self.observed_frames},
                latent_mask=loss_mask,
                eval_mask=latent_mask,
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
        self.lr_scheduler.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        self.lr_scheduler.step()
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
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("lr", self.lr_scheduler.get_last_lr()[0])
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        if dist.is_initialized() and dist.get_rank() == 0:
            postfix = "latest" if self._args.save_latest_only else f"{(self.step):06d}"
            to_save = {bf.join(get_blob_logdir(self._args), f"opt_{postfix}.pt"): self.opt.state_dict()}

            # vmnote: make dir if it doesn't exist
            Path(get_blob_logdir(self._args)).mkdir(parents=True, exist_ok=True)

            for rate, params in zip([0, *self.ema_rate],
                                    [self.master_params, *self.ema_params]):
                filename = f"ema_{rate}_{postfix}.pt" if rate else f"model_{postfix}.pt"
                filepath = bf.join(get_blob_logdir(self._args), filename)
                to_save[filepath] = {
                    "state_dict": self._master_params_to_state_dict(params),
                    "config": self._args.__dict__,
                    "step": self.step
                }

            for path in to_save:
                # backup previous
                if os.path.exists(path) and self._args.save_latest_only:
                    shutil.copy(path, path + '-backup')
            for path, params in to_save.items():
                with bf.BlobFile(path, "wb") as f:
                    th.save(params, f)
            for path in to_save:
                # delete backup
                backup_path = path + '-backup'
                if os.path.exists(backup_path):
                    os.remove(backup_path)
            dist.barrier()
        else:
            postfix = "latest" if self._args.save_latest_only else f"{(self.step):06d}"
            to_save = {bf.join(get_blob_logdir(self._args), f"opt_{postfix}.pt"): self.opt.state_dict()}

            # vmnote: make dir if it doesn't exist
            Path(get_blob_logdir(self._args)).mkdir(parents=True, exist_ok=True)

            for rate, params in zip([0, *self.ema_rate],
                                    [self.master_params, *self.ema_params]):
                filename = f"ema_{rate}_{postfix}.pt" if rate else f"model_{postfix}.pt"
                filepath = bf.join(get_blob_logdir(self._args), filename)
                to_save[filepath] = {
                    "state_dict": self._master_params_to_state_dict(params),
                    "config": self._args.__dict__,
                    "step": self.step
                }

            for path in to_save:
                # backup previous
                if os.path.exists(path) and self._args.save_latest_only:
                    shutil.copy(path, path + '-backup')
            for path, params in to_save.items():
                with bf.BlobFile(path, "wb") as f:
                    th.save(params, f)
            for path in to_save:
                # delete backup
                backup_path = path + '-backup'
                if os.path.exists(backup_path):
                    os.remove(backup_path)

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

    def make_interesting_masks(self, batch):
        n_masks = min(self.n_interesting_masks, len(batch))

        def make_zeros():
            return th.zeros_like(batch[:n_masks, :, :1, :1, :1])

        obs_mask = make_zeros()
        latent_mask = make_zeros()
        kinda_marg_mask = make_zeros()
        n_obs = self.max_frames // 3
        n_latent = self.max_frames - n_obs
        for i in range(n_masks):
            spacing = 1 if n_masks == 1 else int((batch.shape[1] // self.max_frames) ** (i / (n_masks - 1)))
            obs_mask[i, :n_obs * spacing:spacing] = 1.
            latent_mask[i, n_obs * spacing:self.max_frames * spacing:spacing] = 1.
        return {'obs': obs_mask, 'latent': latent_mask, 'kinda_marg': kinda_marg_mask}

    @rng_decorator(seed=0)
    def log_samples(self):
        sample_start = time()
        self.model.eval()
        orig_state_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(copy.deepcopy(self._master_params_to_state_dict(self.ema_params[0])))

        logger.log("sampling...")
        orig_batch = th.cat(self.valid_batches, dim=0).to(dist_util.dev())
        set_masks = self.make_interesting_masks(orig_batch)
        batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask = \
            self.sample_all_masks(orig_batch, set_masks=set_masks)
        sample_fn = (
            self.diffusion.p_sample_loop
        )

        def repeat(t):
            return th.repeat_interleave(t, repeats=self.n_valid_repeats, dim=0)

        batch, orig_batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask = map(
            repeat, [batch, orig_batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask])

        samples = []
        attns = []

        def chunk(t):
            return th.chunk(t, dim=0, chunks=self.n_valid_batches * self.n_valid_repeats)

        for x0, fi, om, lm, kmm in zip(*map(
                chunk, [batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask])):
            s, a = sample_fn(
                self.model,
                x0.shape,
                clip_denoised=True,
                model_kwargs={
                    'frame_indices': fi,
                    'x0': x0, 'obs_mask': om,
                    'latent_mask': lm,
                    'kinda_marg_mask': kmm,
                    'observed_frames': self.observed_frames},
                latent_mask=lm,
                return_attn_weights=True,
            )
            samples.append(s)
            attns.append(a)
        sample = th.cat(samples, dim=0)
        attns = {k: th.cat([a[k] for a in attns], dim=0) for k in attns[0].keys()}

        # visualise the samples -----------------------------------------------
        marked_batch = batch.clone()
        _mark_as_observed(marked_batch)
        vis = sample * latent_mask + marked_batch * obs_mask
        vis_all = th.zeros_like(orig_batch)
        error = latent_mask * (sample - batch)
        error_all = th.zeros_like(orig_batch)
        for b in range(len(batch)):
            is_latent = latent_mask[b, :, 0, 0, 0].bool()
            is_obs = obs_mask[b, :, 0, 0, 0].bool()
            existing_frame_indices = frame_indices[b, is_latent + is_obs]
            vis_all[b, existing_frame_indices] = vis[b, :len(existing_frame_indices)]
            latent_frame_indices = frame_indices[b, is_latent]
            error_all[b, latent_frame_indices] = error[b, is_latent]
        rmse = ((error ** 2).mean() / latent_mask.mean()).sqrt()
        gather_and_log_videos('sample/', vis_all, log_as='array')
        n_samples_with_preset_masks = len(set_masks['obs']) * self.n_valid_repeats
        if n_samples_with_preset_masks > 0:
            gather_and_log_videos('sample/', vis[:n_samples_with_preset_masks], log_as='video')
        gather_and_log_videos('error/', error_all, log_as='array')
        logger.log("sampling complete")
        logger.logkv('timing/sampling_time', time() - sample_start)
        logger.logkv('rmse', rmse.cpu().item())

        # visualise the attn weights ------------------------------------------
        spatial_attn = {k: v for k, v in attns.items() if 'spatial' in k}
        frame_attn = {k: v for k, v in attns.items() if 'temporal' in k}
        for k, v in spatial_attn.items():
            logger.logkv(k, wandb.Image(concat_images_with_padding(v.unsqueeze(1), horizontal=False).cpu()))
        for k, attn in frame_attn.items():
            fig = Figure(figsize=(5, 4.5 * len(batch)))
            canvas = FigureCanvas(fig)
            axes = [fig.add_subplot(len(batch), 1, i + 1) for i in range(len(batch))]
            for fi, attn_matrix, ax in zip(frame_indices.cpu().numpy(), attn.cpu(), axes):
                n_frames = attn_matrix.shape[-1]
                ax.imshow(attn_matrix, vmin=0, cmap='binary_r')
                for axis, set_ticks, set_labels, set_lim in [('x', ax.set_xticks, ax.set_xticklabels, ax.set_xlim),
                                                             ('y', ax.set_yticks, ax.set_yticklabels, ax.set_ylim)]:
                    set_ticks(np.linspace(0, n_frames - 1, n_frames))
                    set_labels(fi)  # (fi if axis == 'x' else fi[::-1])
                    set_lim(-0.5, n_frames - 0.5)
            logger.logkv(k, fig)
        self.model.train()
        self.model.load_state_dict(orig_state_dict)

    def visualise(self):
        batch = th.cat(self.valid_batches)
        batch, obs_mask, latent_mask, kinda_marg_mask = self.sample_all_masks(batch, gather=False)
        vis = th.ones_like(batch)
        vis[obs_mask.expand_as(batch) == 1] = batch[obs_mask.expand_as(batch) == 1]
        for quartile in [0, 1, 2, 3, 3.99]:
            t = th.tensor(self.diffusion.num_timesteps * (quartile / 4)).int()
            xt = self.diffusion.q_sample(batch, t=t)
            vis[latent_mask.expand_as(batch) == 1] = xt[latent_mask.expand_as(batch) == 1]
            gather_and_log_videos(f'visualise/inputs-q{quartile}', vis, log_as='array',
                                  pad_dim_h=4, pad_dim_v=4, pad_val=0, pad_ends=True)
        print(vis.shape, obs_mask.shape, batch.shape)
        red = th.tensor([1., 0., 0.]).view(1, 1, 3, 1, 1)
        blue = th.tensor([0., 1., 0.]).view(1, 1, 3, 1, 1)
        vis = th.ones_like(vis)
        vis_red = vis * red
        vis_blue = vis * blue
        vis[obs_mask.expand_as(batch) == 1] = vis_red[obs_mask.expand_as(batch) == 1]
        vis[latent_mask.expand_as(batch) == 1] = vis_blue[latent_mask.expand_as(batch) == 1]
        gather_and_log_videos('visualise/mask', vis, log_as='array', pad_dim_h=4, pad_dim_v=12, pad_val=0, pad_ends=True)
        logger.dumpkvs()

    def save_masks(self, n_masks):
        # for use with video_nll
        batch = th.zeros(1, self.T, 3, 64, 64)
        obs_indices = []
        lat_indices = []
        for i in range(n_masks):
            with RNG(i):
                batch, obs_mask, latent_mask, kinda_marg_mask = self.sample_all_masks(batch, gather=False)
            obs_indices += [[list(layer.nonzero(as_tuple=True)[0].flatten().numpy())] for layer in obs_mask.flatten(start_dim=2)]
            lat_indices += [[list(layer.nonzero(as_tuple=True)[0].flatten().numpy())] for layer in latent_mask.flatten(start_dim=2)]
        path = f"samples/indices/{self._args.mask_distribution}_{self._args.max_frames}_{self._args.T}_frame_indices.pt"
        th.save((obs_indices, lat_indices), path)


def _mark_as_observed(images, color=[1., -1., -1.]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


def concat_images_with_padding(images, horizontal=True,
                               pad_dim=1, pad_val=0, pad_ends=False):
    """Cocatenates a list (or batched tensor) of CxHxW images, with padding in
    between, for pretty viewing.
    """
    _, h, w = images[0].shape
    pad_h, pad_w = (h, pad_dim) if horizontal else (pad_dim, w)
    padding = th.zeros_like(images[0][:, :pad_h, :pad_w]) + pad_val
    images_with_padding = []
    for image in images:
        images_with_padding.extend([image, padding])
    if pad_ends:
        images_with_padding = [padding, *images_with_padding, padding]
    images_with_padding = images_with_padding[:-1]  # remove final pad
    return th.cat(images_with_padding, dim=2 if horizontal else 1)


def gather_and_log_videos(name, array, log_as='both', pad_dim_h=1, pad_dim_v=1, pad_val=255, pad_ends=False):
    """
    Unnormalises and logs videos given as B x T x C x H x W tensors.
        :`as` can be 'array', 'video', or 'both'
    """
    array = array.to(dist_util.dev())
    array = ((array + 1) * 127.5).clamp(0, 255).to(th.uint8)
    array = array.contiguous()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    gathered_arrays = [th.zeros_like(array) for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather(gathered_arrays, array)  # gather not supported with NCCL
    videos = th.cat([array.cpu() for array in gathered_arrays], dim=0)
    if dist.is_initialized():
        dist.barrier()

    if log_as in ['array', 'both']:
        # log all videos/frames as one single image array
        img = concat_images_with_padding(
            [concat_images_with_padding(vid, horizontal=True, pad_dim=pad_dim_h, pad_val=pad_val, pad_ends=pad_ends) for vid in videos],
            horizontal=False, pad_dim=pad_dim_v, pad_val=pad_val, pad_ends=pad_ends,
        )
        img = img.permute(1, 2, 0).numpy()
        logger.logkv(name + 'array', wandb.Image(img))
    if log_as in ['video', 'both']:
        # log each batch element as its own video
        final_frame = th.zeros_like(videos[0, :1])
        final_frame[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
        for i, video in enumerate(videos):
            logger.logkv(name + f'video-{i}', wandb.Video(th.cat([video, final_frame], dim=0)))


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


def get_blob_logdir(args):
    root_dir = os.environ.get("DIFFUSION_BLOB_LOGDIR", "checkpoints")
    assert os.path.exists(root_dir), "Must create directory 'checkpoints' or specify existing DIFFUSION_BLOB_LOGDIR"
    wandb_id = args.resume_id if len(args.resume_id) > 0 else wandb.run.id
    return os.path.join(root_dir, wandb_id)


def find_resume_checkpoint(args):
    """
    If there are checkpoints saved in get_blob_logdir(), will return the latest one
    """
    if not args.resume_id:
        return
    logdir = get_blob_logdir(args)
    print('looking in', logdir)
    if not os.path.exists(logdir):
        return
    logpath = os.path.join(logdir, 'model_latest.pt')
    if os.path.exists(logpath):
        return logpath
    else:
        logpaths = glob.glob(os.path.join(get_blob_logdir(args), 'model_*.pt'))
        latest_step = -1
        logpath = None
        for d in logpaths:
            step = int(os.path.splitext(d)[0].split('_')[-1])
            if step > latest_step:
                latest_step = step
                logpath = d
        if logpath is not None:
            return logpath


def find_ema_checkpoint(main_checkpoint, step, rate, save_latest_only):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_latest.pt" if save_latest_only else f"ema_{rate}_{(step):06d}.pt"
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
