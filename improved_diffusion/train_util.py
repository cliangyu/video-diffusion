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
        n_valid_batches=1,
        max_frames=10,
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
        self.max_frames = max_frames
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
        self.n_valid_batches = n_valid_batches
        self.valid_batches = [next(self.data)[0][:self.microbatch]
                            for i in range(self.n_valid_batches)]

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

    # def sample_video_mask(self, data, mask_type, min_ones, max_ones, exclude=None, set_mask=()):  # max_ones is max_zeros if mask_type=='marg'
    #     like = data[len(set_mask):, :, :1, :1, :1]  # B x T x 1 x 1 x 1
    #     B, T, *_ = like.shape
    #     if exclude is None:
    #         exclude = th.zeros_like(like)
    #     else:
    #         exclude = exclude[len(set_mask):]    # TODO flip definitions of marg masks to be latent_masks
    #     if mask_type == 'zero':
    #         mask = th.zeros_like(like)
    #     elif mask_type in ['obs', 'marg']:
    #         mask = th.zeros_like(like)
    #         for row, row_exclude in zip(mask, exclude):
    #             print(row_exclude.sum().cpu().item(), max_ones, min_ones, mask_type)
    #             n_ones = np.random.randint(min_ones, max_ones-row_exclude.sum().cpu().item()+1)
    #             while row.sum() < n_ones:
    #                 s = min(np.random.randint(1, n_ones+1), n_ones-row.sum().cpu().int().item())
    #                 max_scale = T / (s-0.999)
    #                 scale = np.exp(np.random.rand() * np.log(max_scale))
    #                 pos = np.random.rand() * (T - scale*(s-1))
    #                 indices = [int(pos+i*scale) for i in range(s)]
    #                 row[indices] = 1.
    #                 row[row_exclude==1] = 0.
    #         if mask_type == 'marg':
    #             mask = 1 - mask
    #     if exclude is not None:
    #         mask = mask * (1 - exclude)
    #     if len(set_mask) > 0:
    #         mask = th.cat([set_mask, mask], dim=0)
    #     return mask

    # def sample_all_masks(self, batch, set_masks={'obs': (), 'marg': (), 'zero': ()}):
    #     N = self.max_frames
    #     pt, ft = ('marg', 'zero') if self.do_inefficient_marg else ('zero', 'marg')
    #     partly_marg_mask = self.sample_video_mask(batch, pt, min_ones=1, max_ones=N, set_mask=set_masks[pt])
    #     latent_mask = 1 - partly_marg_mask
    #     fully_marg_mask = self.sample_video_mask(batch, ft, min_ones=1, max_ones=N, exclude=1-latent_mask, set_mask=set_masks[ft])
    #     latent_mask = latent_mask * (1-fully_marg_mask)
    #     obs_mask = self.sample_video_mask(batch, 'obs', min_ones=0, max_ones=N, exclude=1-latent_mask, set_mask=set_masks['obs'])
    #     latent_mask = latent_mask * (1 - obs_mask)
    #     # delete as many frames as possible fiven fully_marg_mask
    #     not_fully_marg_mask, (batch, obs_mask, partly_marg_mask, latent_mask), frame_indices =\
    #         self.gather_unmasked_elements(
    #             (1-fully_marg_mask), [batch, obs_mask, partly_marg_mask, latent_mask]
    #     )
    #     fully_marg_mask = 1 - not_fully_marg_mask
    #     # print('latent', latent_mask.flatten(start_dim=1).sum(dim=1),
    #     #       '\nobs', obs_mask.flatten(start_dim=1).sum(dim=1),
    #     #       '\npartly', partly_marg_mask.flatten(start_dim=1).sum(dim=1),
    #     #       '\nfully', fully_marg_mask.flatten(start_dim=1).sum(dim=1))
    #     return batch, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, latent_mask

    def sample_some_indices(self, max_indices, T):
        s = th.randint(low=1, high=max_indices+1, size=())
        max_scale = T / (s-0.999)
        scale = np.exp(np.random.rand() * np.log(max_scale))
        pos = th.rand(()) * (T - scale*(s-1))
        return [int(pos+i*scale) for i in range(s)]

    def sample_all_masks(self, batch, set_masks={'obs': (), 'marg': (), 'zero': ()}):
        p_observed_latent_marg = th.tensor([0.33, 0.33, 0.33] if self.do_inefficient_marg else [0.5, 0.5, 0])
        N = self.max_frames
        B, T, *_ = batch.shape
        masks = {k: th.zeros_like(batch[:, :, :1, :1, :1]) for k in ['obs', 'latent', 'kinda_marg']}
        for obs_row, latent_row, marg_row in zip(*[masks[k] for k in ['obs', 'latent', 'kinda_marg']]):
            latent_row[self.sample_some_indices(max_indices=N, T=T)] = 1.
            while True:
                mask_i = th.distributions.Categorical(probs=p_observed_latent_marg).sample()
                mask = [obs_row, latent_row, marg_row][mask_i]
                indices = th.tensor(self.sample_some_indices(max_indices=N, T=T))
                taken = (obs_row[indices] + latent_row[indices] + marg_row[indices]).view(-1)
                indices = indices[taken==0]  # remove indices that are already used in a mask
                if len(indices) > N - sum(obs_row) - sum(latent_row) - sum(marg_row):
                    break
                mask[indices] = 1.
        # TODO fix rest of code to get rid of the bullshit below
        obs_mask = masks['obs']
        partly_marg_mask = masks['kinda_marg']
        fully_marg_mask = 1 - masks['latent'] - masks['obs'] - masks['kinda_marg']
        latent_mask = masks['latent']
        n_set = len(set_masks['obs'])
        if n_set > 0:
            obs_mask[:n_set] = set_masks['obs']
            fully_marg_mask[:n_set] = set_masks['marg']
            partly_marg_mask[:n_set] = 0.
            latent_mask = 1 - obs_mask - fully_marg_mask - partly_marg_mask
        not_fully_marg_mask, (batch, obs_mask, partly_marg_mask, latent_mask), frame_indices =\
            self.gather_unmasked_elements(
                (1-fully_marg_mask), [batch, obs_mask, partly_marg_mask, latent_mask]
        )
        fully_marg_mask = 1 - not_fully_marg_mask
        return batch, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, latent_mask

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

    def make_interesting_masks(self, batch, max_obs_latent=10):
        n_masks = 3
        masks = {'zero': th.zeros_like(batch[:n_masks, :, :1, :1, :1])}
        n_obs = 3
        max_latent = max_obs_latent - n_obs
        masks['obs'] = masks['zero'].clone()
        masks['obs'][:, 1:1+n_obs] = 1.
        masks['marg'] = 1 - masks['obs']
        try:
            masks['marg'][0, 1+n_obs:1+n_obs+max_latent:] = 0.
            masks['marg'][1, 1+n_obs:1+n_obs+max_latent*2:2] = 0.
            masks['marg'][2, 1+n_obs:1+n_obs+max_latent*4:4] = 0.
        except IndexError:
            assert len(masks) < n_masks
        return masks

    @rng_decorator(seed=0)
    def log_samples(self):
        sample_start = time()
        self.model.eval()
        logger.log("sampling...")
        orig_batch = th.cat(self.valid_batches, dim=0).to(dist_util.dev())
        set_masks = self.make_interesting_masks(orig_batch)
        batch, frame_indices, obs_mask, partly_marg_mask, fully_marg_mask, dynamics_mask = \
            self.sample_all_masks(orig_batch, set_masks=set_masks)

        img_size = batch.shape[-1]
        # copied from scripts/image_sample.py ---------------------------------
        sample_fn = (
            self.diffusion.p_sample_loop
        )
        samples = []
        chunk_kwargs = dict(dim=0, chunks=self.n_valid_batches)
        for fi, x0, om, pmm, fmm in zip(
                th.chunk(frame_indices, **chunk_kwargs), th.chunk(batch, **chunk_kwargs),
                th.chunk(obs_mask, **chunk_kwargs), th.chunk(partly_marg_mask, **chunk_kwargs),
                th.chunk(fully_marg_mask, **chunk_kwargs)):
            samples.append(sample_fn(
                self.model,
                x0.shape,
                clip_denoised=True,
                model_kwargs={
                    'frame_indices': fi,
                    'x0': x0, 'obs_mask': om,
                    'partly_marg_mask': pmm,
                    'fully_marg_mask': fmm},
                dynamics_mask=dynamics_mask,
            ))
        sample = th.cat(samples, dim=0)

        # ---------------------------------------------------------------------
        batch_vis = th.zeros_like(orig_batch)
        batch_is_latent = dynamics_mask.view(sample.shape[:2]).bool()
        batch_is_obs = obs_mask.view(sample.shape[:2]).bool()
        tinted_batch = orig_batch.clone()
        tinted_batch[:, :, :1] = 0   # mutilate observed frames
        error = th.zeros_like(orig_batch)
        rmse = 0.
        for vis, error_row, is_latent, is_obs, frame_indices_element, data_element, tinted_element, sampled_element in zip(
                batch_vis, error, batch_is_latent, batch_is_obs, frame_indices, orig_batch, tinted_batch, sample
        ):
            obs_indices = frame_indices_element[is_obs]
            vis[obs_indices] = tinted_element[obs_indices]
            latent_indices = frame_indices_element[is_latent]
            vis[latent_indices] = sampled_element[is_latent]
            error_row[latent_indices] = sampled_element[is_latent] - data_element[latent_indices]
            rmse += (error_row[latent_indices]**2).mean().sqrt() / len(orig_batch)
        gather_and_log_videos('sample', batch_vis)
        gather_and_log_videos('error/sample', error)
        logger.log("sampling complete")
        logger.logkv('sampling_time', time()-sample_start)
        logger.logkv('rmse', rmse.cpu().item())
        self.model.train()


def concat_images_with_padding(images, horizontal=True, pad_dim=1, pad_val=0):
    """Cocatenates a list (or batched tensor) of CxHxW images, with padding in
    between, for pretty viewing.
    """
    _, h, w = images[0].shape
    pad_h, pad_w = (h, pad_dim) if horizontal else (pad_dim, w)
    padding = th.zeros_like(images[0][:, :pad_h, :pad_w]) + pad_val
    images_with_padding = []
    for image in images:
        images_with_padding.extend([image, padding])
    images_with_padding = images_with_padding[:-1]   # remove final pad
    return th.cat(images_with_padding, dim=2 if horizontal else 1)


def gather_and_log_videos(name, array):
    """
    Unnormalises and logs videos given as B x T x C x H x W tensors.
    """
    array = array.cuda()
    array = ((array + 1) * 127.5).clamp(0, 255).to(th.uint8)
    array = array.contiguous()
    gathered_arrays = [th.zeros_like(array) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_arrays, array)  # gather not supported with NCCL
    videos = th.cat([array.cpu() for array in gathered_arrays], dim=0)
    dist.barrier()

    final_frames = th.zeros_like(videos[:, :1])
    final_frames[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
    videos = th.cat([videos, final_frames], dim=1)
    merged_videos = th.stack(
        [concat_images_with_padding(frames, horizontal=False, pad_dim=2)
         for frames in videos.transpose(1, 0)]
    )
    logger.logkv(name, wandb.Video(merged_videos))
    logger.logkv(
        name+'-flat',
        wandb.Image(concat_images_with_padding(merged_videos[:-1], horizontal=True, pad_dim=1).permute(1, 2, 0).numpy())
    )


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

# def inclusice_randint(l, h=None):
#     low, high = (0, l) if h is None else (l, h)
#     return low if low == high else np.random.randint(low, high+1)
