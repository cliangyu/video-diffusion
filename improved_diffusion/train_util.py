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
        self.T = T
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
        self.n_valid_repeats = n_valid_repeats
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

    def sample_some_indices(self, max_indices, T):
        s = th.randint(low=1, high=max_indices+1, size=())
        max_scale = T / (s-0.999)
        scale = np.exp(np.random.rand() * np.log(max_scale))
        pos = th.rand(()) * (T - scale*(s-1))
        indices = [int(pos+i*scale) for i in range(s)]
        # do some recursion if we have somehow failed to satisfy the consrtaints
        if all(i<T and i>=0 for i in indices):
            return indices
        else:
            print('warning: sampled indices', [int(pos+i*scale) for i in range(s)], 'trying again')
            return self.sample_some_indices(max_indices, T)

    def sample_all_masks(self, batch, set_masks={'obs': (), 'latent': (), 'kinda_marg': ()}):
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
                indices = indices[taken == 0]  # remove indices that are already used in a mask
                if len(indices) > N - sum(obs_row) - sum(latent_row) - sum(marg_row):
                    break
                mask[indices] = 1.
        if len(set_masks['obs']) > 0:
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        represented_mask = (masks['obs'] + masks['latent'] + masks['kinda_marg']).clip(max=1)
        represented_mask, (batch, obs_mask, latent_mask, kinda_marg_mask), frame_indices =\
            self.gather_unmasked_elements(
                represented_mask, (batch, masks['obs'], masks['latent'], masks['kinda_marg'])
            )
        return batch, frame_indices, obs_mask, latent_mask, kinda_marg_mask

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

            t_0 = time()
            batch, cond = next(self.data)

            self.run_step(batch, cond)
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
                logger.logkv('timing/time_between_samples', time()-last_sample_time)
                last_sample_time = time()
            self.step += 1
            if self.step == 1:
                gather_and_log_videos('data/', batch, log_as='both')
                assert len(batch[0]) == self.T
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
            micro = batch[i:i + self.microbatch].to(dist_util.dev())
            micro, frame_indices, obs_mask, latent_mask, kinda_marg_mask = self.sample_all_masks(micro)
            micro_cond = {
                k: v[i:i+self.microbatch].to(dist_util.dev())
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
                              'latent_mask': latent_mask, 'kinda_marg_mask': kinda_marg_mask,
                              'x0': micro},
                latent_mask=latent_mask,
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

    def make_interesting_masks(self, batch):
        n_masks = min(3, len(batch))
        def make_zeros():
            return th.zeros_like(batch[:n_masks, :, :1, :1, :1])
        obs_mask = make_zeros()
        latent_mask = make_zeros()
        kinda_marg_mask = make_zeros()
        n_obs = self.max_frames // 3
        n_latent = self.max_frames - n_obs
        for i in range(n_masks):
            spacing = 1 if n_masks == 1 else int((batch.shape[1] // self.max_frames)**(i/(n_masks-1)))
            obs_mask[i, :n_obs*spacing:spacing] = 1.
            latent_mask[i, n_obs*spacing:self.max_frames*spacing:spacing] = 1.
        return {'obs': obs_mask, 'latent': latent_mask, 'kinda_marg': kinda_marg_mask}

    @rng_decorator(seed=0)
    def log_samples(self):
        sample_start = time()
        self.model.eval()
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
            return th.chunk(t, dim=0, chunks=self.n_valid_batches*self.n_valid_repeats)
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
                    'kinda_marg_mask': kmm},
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
        vis = sample*latent_mask + marked_batch*obs_mask
        vis_all = th.zeros_like(orig_batch)
        error = latent_mask * (sample - batch)
        error_all = th.zeros_like(orig_batch)
        for b in range(len(batch)):
            is_latent = latent_mask[b, :, 0, 0, 0].bool()
            is_obs = obs_mask[b, :, 0, 0, 0].bool()
            existing_frame_indices = frame_indices[b, is_latent+is_obs]
            vis_all[b, existing_frame_indices] = vis[b, :len(existing_frame_indices)]
            latent_frame_indices = frame_indices[b, is_latent]
            error_all[b, latent_frame_indices] = error[b, is_latent]
        rmse = ((error**2).mean()/latent_mask.mean()).sqrt()
        gather_and_log_videos('sample/', vis_all, log_as='array')
        n_samples_with_preset_masks = len(set_masks['obs']) * self.n_valid_repeats
        gather_and_log_videos('sample/', vis[:n_samples_with_preset_masks], log_as='video')
        gather_and_log_videos('error/', error_all, log_as='array')
        logger.log("sampling complete")
        logger.logkv('timing/sampling_time', time()-sample_start)
        logger.logkv('rmse', rmse.cpu().item())

        # visualise the attn weights ------------------------------------------
        def reshape_attn(attn, to):
            b, q, k = attn.shape
            assert q == k
            d = int((q // self.max_frames)**0.5)
            attn = attn.view(b, self.max_frames, d, d, self.max_frames, d, d)
            if to == 'frame':
                attn = attn.sum(dim=(2, 3, 5, 6)).view(b, self.max_frames, self.max_frames)
            elif to == 'spatial':
                frames = list(range(self.max_frames))
                attn = attn[:, frames, :, :, frames, :, :].sum(dim=0).view(b, d*d, d*d)
            else:
                raise NotImplementedError()
            return attn
        spatial_attn = {'spatial-'+k: reshape_attn(v, 'spatial') for k, v in attns.items()}
        frame_attn = {'frame-'+k: reshape_attn(v, 'frame') for k, v in attns.items()}
        for k, v in spatial_attn.items():
            logger.logkv(k, wandb.Image(concat_images_with_padding(v.unsqueeze(1), horizontal=False).cpu()))
        for k, attn in frame_attn.items():
            fig = Figure()
            canvas = FigureCanvas(fig)
            axes = [fig.add_subplot(len(batch), 1, i+1) for i in range(len(batch))]
            for fi, attn_matrix, ax in zip(frame_indices.cpu().numpy(), attn.cpu(), axes):
                n_frames = attn_matrix.shape[-1]
                ax.imshow(attn_matrix, vmin=0, cmap='binary_r')
                for axis, set_ticks, set_labels, set_lim in [('x', ax.set_xticks, ax.set_xticklabels, ax.set_xlim),
                                                             ('y', ax.set_yticks, ax.set_yticklabels, ax.set_ylim)]:
                    set_ticks(np.linspace(0, n_frames-1, n_frames))
                    set_labels(fi if axis == 'x' else fi[::-1])
                    set_lim(-0.5, n_frames-0.5)
            logger.logkv(k, fig)
        self.model.train()


def _mark_as_observed(images, color=[1., -1., -1.]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


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


def gather_and_log_videos(name, array, log_as='both'):
    """
    Unnormalises and logs videos given as B x T x C x H x W tensors.
        :`as` can be 'array', 'video', or 'both'
    """
    array = array.to(dist_util.dev())
    array = ((array + 1) * 127.5).clamp(0, 255).to(th.uint8)
    array = array.contiguous()
    gathered_arrays = [th.zeros_like(array) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_arrays, array)  # gather not supported with NCCL
    videos = th.cat([array.cpu() for array in gathered_arrays], dim=0)
    dist.barrier()

    if log_as in ['array', 'both']:
        # log all videos/frames as one single image array
        img = concat_images_with_padding(
            [concat_images_with_padding(vid, horizontal=True, pad_dim=1, pad_val=255) for vid in videos],
            horizontal=False, pad_dim=2, pad_val=255
        )
        img = img.permute(1, 2, 0).numpy()
        logger.logkv(name+'array', wandb.Image(img))
    if log_as in ['video', 'both']:
        # log each batch element as its own video
        final_frame = th.zeros_like(videos[0, :1])
        final_frame[..., ::2, 1::2] = 255  # checkerboard pattern to mark the end
        for i, video in enumerate(videos):
            logger.logkv(name+f'video-{i}', wandb.Video(th.cat([video, final_frame], dim=0)))


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
