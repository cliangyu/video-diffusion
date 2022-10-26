import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (SiLU, avg_pool_nd, checkpoint, conv_nd, frame_embedding,
                 linear, normalization, timestep_embedding, zero_module)


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second
    argument."""
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input."""
    def forward(self,
                x,
                emb,
                attn_mask,
                T=1,
                frame_indices=None,
                attn_weights_list=None):
        for layer in self:
            kwargs = {}
            if isinstance(layer, TimestepBlock):
                kwargs[
                    'emb'] = emb  # vmnote: 'emb' here is the timestep embedding
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs['temb'] = emb
                kwargs['attn_mask'] = attn_mask
                kwargs['T'] = T
                kwargs['attn_weights_list'] = attn_weights_list
                kwargs['frame_indices'] = frame_indices
            x = layer(x, **kwargs)
        return x


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              channels,
                              channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# Resblocks are a replacement for convolutional layers.
class ResBlock(TimestepBlock):
    """A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims,
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           3,
                                           padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels,
                                           1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# vmnote: this is new, study it
class FactorizedAttentionBlock(nn.Module):
    """Loosely based on CSDI's factorized attention for time-series data.

    https://openreview.net/pdf?id=VzuIzbRDrum
    """
    def __init__(
        self,
        channels,
        num_heads,
        use_rpe_net,
        time_embed_dim=None,
        use_checkpoint=False,
        temporal_augment_type='none',
        bucket_params=None,
        allow_interactions_between_padding=False,
    ):
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            use_rpe_q=False,
            use_rpe_k=False,
            use_rpe_v=False,
        )
        self.temporal_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            bucket_params=bucket_params,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
            allow_interactions_between_padding=
            allow_interactions_between_padding,
        )

    def forward(self,
                x,
                attn_mask,
                temb,
                T,
                attn_weights_list=None,
                frame_indices=None):
        BT, C, H, W = x.shape
        B = BT // T
        # reshape to have T in the last dimension becuase that's what we attend over
        x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
        x = x.reshape(B, H * W, C, T)
        x = self.temporal_attention(
            x,
            temb,
            frame_indices,  # B x T
            attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2),  # B x T
            attn_weights_list=None
            if attn_weights_list is None else attn_weights_list['temporal'],
        )

        # Now we attend over the spatial dimensions by reshaping the input
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
        x = x.reshape(B, T, C, H * W)
        x = self.spatial_attention(
            x,
            temb,  # TODO reshape temb?
            frame_indices=None,
            attn_weights_list=None
            if attn_weights_list is None else attn_weights_list['spatial'],
        )
        x = x.reshape(BT, C, H, W)
        return x


class RPENet(nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.0
        self.out.bias.data *= 0.0
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, temb, relative_distances):
        distance_embs = th.stack(
            [
                th.log(1 + (relative_distances).clamp(min=0)),
                th.log(1 + (-relative_distances).clamp(min=0)),
                (relative_distances == 0).float(),
            ],
            dim=-1,
        )  # BxTxTx3
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(temb).view(
            B, T, 1, C) + self.embed_distances(distance_embs)  # B x T x T x C
        return self.out(self.silu(emb)).view(*relative_distances.shape,
                                             self.num_heads,
                                             self.channels // self.num_heads)


class RPE(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self,
                 channels,
                 num_heads,
                 bucket_params,
                 time_embed_dim,
                 use_rpe_net=False):
        """This module handles the relative positional encoding.

        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
            bucket_params (dict): Parameters for the buckets. It should be a dictionary with
                the keys ["alpha", "beta", "gamma"] as defined in eq. 18 of https://arxiv.org/pdf/2107.14222.pdf
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.alpha = bucket_params['alpha']
        self.beta = bucket_params['beta']
        self.gamma = bucket_params['gamma']
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = nn.Parameter(
                th.zeros(2 * self.beta + 1, self.num_heads, self.head_dim))

    def get_bucket_ids(self, pairwise_distances):
        # Based on Eq. 18 of https://arxiv.org/pdf/2107.14222.pdf
        bucket_ids = pairwise_distances.clone()
        mask = bucket_ids.abs() > self.alpha
        if mask.sum() > 0:
            coef = th.log(bucket_ids[mask].abs() / self.alpha) / np.log(
                self.gamma / self.alpha)
            bucket_ids[mask] = th.minimum(
                th.tensor(self.beta), self.alpha + coef *
                (self.beta - self.alpha)).int() * th.sign(bucket_ids[mask])
        return bucket_ids

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            bucket_ids = self.get_bucket_ids(pairwise_distances)
            return self.lookup_table_weight[bucket_ids]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == 'qk':
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == 'v':
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f'Unexpected RPE attention mode: {mode}')

    def forward_qk(self, qk, pairwise_distances, temb):
        # qv is either of q or k and has shape BxDxHxTx(C/H)
        # Output shape should be # BxDxHxTxT
        # bucket_ids: BxTxT
        R = self.get_R(pairwise_distances, temb)
        return th.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
            'bdhtf,btshf->bdhts',
            qk,
            R  # BxDxHxTxT
        )

    def forward_v(self, attn, pairwise_distances, temb):
        # attn has shape BxDxHxTxT
        # Output shape should be # BxDxHxYx(C/H)
        # bucket_ids: BxTxT
        R = self.get_R(pairwise_distances, temb)
        th.einsum('bdhts,btshf->bdhtf', attn, R)
        return th.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
            'bdhts,btshf->bdhtf',
            attn,
            R  # BxDxHxTxT
        )

    def forward_safe_qk(self, x, pairwise_distances, temb):
        R = self.get_R(pairwise_distances, temb)
        B, T, _, H, F = R.shape
        D = x.shape[1]
        res = x.new_zeros(B, D, H, T, T)  # attn shape
        for b in range(B):
            for d in range(D):
                for h in range(H):
                    for i in range(T):
                        for j in range(T):
                            res[b, d, h, i, j] = x[b, d, h, i].dot(R[b, i, j,
                                                                     h])
        return res


class RPEAttention(nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    """Attention with image relative position encoding."""
    def __init__(
        self,
        channels,
        num_heads,
        use_checkpoint=False,
        bucket_params=None,
        time_embed_dim=None,
        use_rpe_net=None,
        use_rpe_q=True,
        use_rpe_k=True,
        use_rpe_v=True,
        allow_interactions_between_padding=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim**-0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)
        self.allow_interactions_between_padding = allow_interactions_between_padding

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert bucket_params is not None
            assert use_rpe_net is not None
            assert ('alpha' in bucket_params and 'beta' in bucket_params
                    and 'gamma' in bucket_params)

        # relative position encoding
        self.rpe_q = (RPE(
            channels=channels,
            num_heads=num_heads,
            bucket_params=bucket_params,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
        ) if use_rpe_q else None)
        self.rpe_k = (RPE(
            channels=channels,
            num_heads=num_heads,
            bucket_params=bucket_params,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
        ) if use_rpe_k else None)
        self.rpe_v = (RPE(
            channels=channels,
            num_heads=num_heads,
            bucket_params=bucket_params,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
        ) if use_rpe_v else None)

    def forward(self,
                x,
                temb,
                frame_indices,
                attn_mask=None,
                attn_weights_list=None):
        out, attn = checkpoint(
            self._forward,
            (x, temb, frame_indices, attn_mask),
            self.parameters(),
            self.use_checkpoint,
        )
        if attn_weights_list is not None:
            B, D, C, T = x.shape
            attn_weights_list.append(attn.detach().view(
                B * D, -1, T, T).mean(dim=1).abs(
                ))  # this is for logging purposes to visualize attn weights
        return out

    def _forward(self, x, temb, frame_indices, attn_mask):
        B, D, C, T = x.shape
        x = x.reshape(B * D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = th.einsum('BDCT -> BDTC', x)  # BxDxTxC
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads,
                                  C // self.num_heads)
        qkv = th.einsum('BDTtHF -> tBDHTF', qkv)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v shapes: BxDxHxTx(C/H)

        q *= self.scale

        attn = q @ k.transpose(-2, -1)  # BxDxHxTxT

        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = frame_indices.unsqueeze(
                -1) - frame_indices.unsqueeze(-2)  # BxTxT
        # pairwise_distances[b, i, j] = frame_indices[b, i] - frame_indices[b, j]

        # w1 = self.rpe_k(q, pairwise_distances, mode="qk")
        # w2 = self.rpe_k.forward_safe_qk(q, pairwise_distances)
        # assert th.abs(w2 - w1).max() < 1e-6

        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, temb=temb, mode='qk')

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale,
                               pairwise_distances,
                               temb=temb,
                               mode='qk').transpose(-1, -2)

        def softmax(w, attn_mask, allow_interactions_between_padding):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(
                    B, T, 1
                )  # locations in video attend to all other locations in video
                # allowed_interactions[:, range(T), range(T)] = 1.
                if allow_interactions_between_padding:
                    allowed_interactions += (1 - attn_mask.view(B, 1, T)) * (
                        1 - attn_mask.view(B, T, 1))
                else:
                    allowed_interactions[:, range(T), range(T)] = 1.0
                inf_mask = 1 - allowed_interactions
                inf_mask[inf_mask == 1] = th.inf
                w = w - inf_mask.view(B, 1, 1, T, T)  # BxDxHxTxT
            return th.softmax(w.float(), dim=-1).type(w.dtype)

        attn = softmax(attn, attn_mask,
                       self.allow_interactions_between_padding)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, temb=temb, mode='v')

        out = th.einsum('BDHTF -> BDTHF', out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = th.einsum('BDTC -> BDCT', x)
        return x, attn


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_spatial_encoding=False,
        image_size=None,
        temporal_augment_type=None,
        use_rpe_net=False,
        bucket_params=None,
        allow_interactions_between_padding=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.n_blocks_before_attn = None
        self.input_blocks = nn.ModuleList([
            TimestepEmbedAttnThingsSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if ds in attention_resolutions and self.n_blocks_before_attn is None:  # first res block
                    self.n_blocks_before_attn = len(self.input_blocks)
                    first_attn_ds = ds
                    first_attn_ch = ch
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            temporal_augment_type=temporal_augment_type,
                            bucket_params=bucket_params,
                            allow_interactions_between_padding=
                            allow_interactions_between_padding,
                        ))
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(
                        Downsample(ch, conv_resample, dims=dims)))
                input_block_chans.append(ch)
                ds *= 2

        if self.n_blocks_before_attn is None:
            self.n_blocks_before_attn = len(self.input_blocks)
            first_attn_ds = ds
            first_attn_ch = ch

        if use_spatial_encoding:
            first_attn_res = image_size // first_attn_ds
            # vmnote: added by will to give the model information about spatial position of pixel within a frame. Note that this is differnet from timestep embeddings in tha this has learned parameters and isn't a function of i/j location, while timestep embeddings are deterministic transforms of diffustion step t
            self.spatial_encoding = nn.Parameter(
                th.randn(1, first_attn_ch, first_attn_res, first_attn_res),
                requires_grad=True,
            )
        else:
            self.spatial_encoding = None

        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            FactorizedAttentionBlock(
                ch,
                num_heads=num_heads,
                use_rpe_net=use_rpe_net,
                time_embed_dim=time_embed_dim,
                use_checkpoint=use_checkpoint,
                temporal_augment_type=temporal_augment_type,
                bucket_params=bucket_params,
                allow_interactions_between_padding=
                allow_interactions_between_padding,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(
                        ),  # vmnote: different than input block
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            temporal_augment_type=temporal_augment_type,
                            bucket_params=bucket_params,
                            allow_interactions_between_padding=
                            allow_interactions_between_padding,
                        ), )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(
                    TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """Get the dtype used by the torso of the model."""
        return next(self.input_blocks.parameters()).dtype

    def forward(
        self,
        x,
        timesteps,
        y=None,
        attn_mask=None,
        T=1,
        return_attn_weights=False,
        frame_indices=None,
        **kwargs,
    ):
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), 'must specify y if and only if the model is class-conditional'

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0], )
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        attns = ({
            'spatial': [],
            'temporal': [],
            'mixed': []
        } if return_attn_weights else None)
        for layer, module in enumerate(
                self.input_blocks
        ):  # add frame embedding after first input block
            h = module(
                h,
                emb,
                attn_mask,
                T=T,
                attn_weights_list=attns,
                frame_indices=frame_indices,
            )
            hs.append(h)
            if layer + 1 == self.n_blocks_before_attn:
                h = self.add_positional_encodings(h,
                                                  frame_indices=frame_indices)
        h = self.middle_block(h,
                              emb,
                              attn_mask,
                              T=T,
                              attn_weights_list=attns,
                              frame_indices=frame_indices)
        for module in self.output_blocks:
            cat_in = th.cat(
                [h, hs.pop()],
                dim=1)  # vmnote: concatenation here is a skip connection
            h = module(
                cat_in,
                emb,
                attn_mask,
                T=T,
                attn_weights_list=attns,
                frame_indices=frame_indices,
            )
        h = h.type(x.dtype)
        out = self.out(h)
        return out, attns

    def add_positional_encodings(self, h):
        if self.spatial_encoding is not None:
            h = h + self.spatial_encoding
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0], )
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result['down'].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result['middle'] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result['up'].append(h.type(x.dtype))
        return result


class UNetVideoModel(UNetModel):
    def __init__(
        self,
        T,
        use_frame_encoding,
        cross_frame_attention,
        enforce_position_invariance,
        *args,
        **kwargs,
    ):
        self.T = T
        self.use_frame_encoding = use_frame_encoding
        self.cross_frame_attention = cross_frame_attention
        self.enforce_position_invariance = enforce_position_invariance
        super().__init__(
            *args,
            **kwargs,
        )

    def forward(self, x, timesteps, frame_indices=None, **kwargs):
        B, T, C, H, W = x.shape
        if frame_indices is None:
            frame_indices = th.arange(0, T,
                                      device=x.device).view(1, T).expand(B, T)
        x = x.view(B * T, C, H, W)
        timesteps = timesteps.reshape(B * T)
        out, attn = super().forward(
            x,
            timesteps,
            frame_indices=frame_indices,
            T=T if self.cross_frame_attention else 1,
            **kwargs,
        )
        return out.view(B, T, self.out_channels, H, W), attn

    def add_positional_encodings(self, h, frame_indices):
        h = super().add_positional_encodings(h)
        if not self.use_frame_encoding:
            return h
        B, T = frame_indices.shape
        BT, C, H, W = h.shape
        assert BT == B * T
        max_period = self.T * 10
        if self.enforce_position_invariance:
            frame_indices = frame_indices.float() - frame_indices.float().mean(
                dim=1, keepdim=True)
        emb = frame_embedding(frame_indices, C, max_period=max_period)
        return h + emb.view(BT, C, 1, 1)


class CondMargVideoModel(
        UNetVideoModel
):  # TODO could generalise to derive similar class for image model
    def __init__(self, cond_emb_type, **kwargs):
        if 'channel' in cond_emb_type:  # only thing for which kinda_marg works
            kwargs['in_channels'] += 2
        elif 'duplicate' in cond_emb_type or 'all' in cond_emb_type:
            kwargs['in_channels'] *= 2
        elif cond_emb_type == 't=0':
            pass
        else:
            raise NotImplementedError
        super().__init__(**kwargs)
        if cond_emb_type == 'channel-initzero':
            self.input_blocks[0][0].weight.data[:, 3] = 0.0
        if cond_emb_type in ['duplicate-initzero', 'all-initzero']:
            self.input_blocks[0][0].weight.data[:, 3:] = self.input_blocks[0][
                0].weight.data[:, :3]
        self.cond_emb_type = cond_emb_type.replace('-initzero', '')

    def forward(self, x, x0, obs_mask, latent_mask, kinda_marg_mask, timesteps,
                **kwargs):
        B, T, C, H, W = x.shape
        timesteps = timesteps.view(B, 1).expand(B, T)
        anything_mask = (obs_mask + latent_mask + kinda_marg_mask).clip(max=1)
        if self.cond_emb_type == 'channel':
            indicator_template = th.ones_like(x[:, :, :1, :, :])
            obs_indicator = indicator_template * obs_mask
            kinda_marg_indicator = indicator_template * kinda_marg_mask
            observed_dict = {
                'x_0': x0,
                'x_t': x,
                'x_t_minus_1': kwargs['x_t_minus_1'],
                'x_random': kwargs['x_random'] if self.training else None,
                'hybrid': kwargs['hybrid'] if self.training else None,
            }
            if 'hybrid' in kwargs['observed_frames']:
                threshold = int(kwargs['observed_frames'].split('_')[-1])
                fully_diffusion_mask = ((
                    timesteps < threshold)[:, :, None, None,
                                           None].expand(obs_mask.shape).int())
                observed_frames = kwargs[
                    'x_t_minus_1'] * fully_diffusion_mask + kwargs[
                        'hybrid'] * (1 - fully_diffusion_mask)
            else:
                observed_frames = observed_dict[kwargs['observed_frames']]
            x = th.cat(
                [
                    x * latent_mask + observed_frames * obs_mask + x *
                    (1 - anything_mask),
                    obs_indicator,
                    kinda_marg_indicator,
                ],
                dim=2,
            )
            # if 'x_t_minus_1' in kwargs:
            #     del kwargs['x_t_minus_1']
            # if 'x_random' in kwargs:
            #     del kwargs['x_random']
            # if 'random_t' in kwargs:
            #     random_t = kwargs['random_t']
            #     del kwargs['random_t']
            timestamp_dict = {
                'x_0':
                th.zeros_like(timesteps[:, 0]),
                'x_t':
                timesteps[:, 0].detach().clone(),
                'x_t_minus_1':
                timesteps[:, 0].detach().clone() - 1,
                'x_random':
                kwargs['random_t'].detach().clone() if self.training else None,
            }
            if 'hybrid' in kwargs['observed_frames']:
                threshold = int(kwargs['observed_frames'].split('_')[-1])
                fully_diffusion_mask = (timesteps < threshold).int()
                timesteps_obs = (
                    fully_diffusion_mask *
                    timestamp_dict['x_t_minus_1'].unsqueeze(-1) +
                    (1 - fully_diffusion_mask) *
                    th.ones_like(timesteps[:, 0]).unsqueeze(-1) * threshold)
            else:
                timesteps_obs = timestamp_dict[
                    kwargs['observed_frames']].expand(T, B).T
            timesteps = timesteps_obs * obs_mask.view(
                B, T) + timesteps * (1 - obs_mask.view(B, T))
        elif self.cond_emb_type in ['duplicate', 'all']:
            x = th.cat(
                [x * latent_mask + x * (1 - anything_mask), x0 * obs_mask],
                dim=2)
        elif self.cond_emb_type in ['t=0', 'all']:
            timesteps[obs_mask.view(B, T) == 1] = -1  # TODO
        else:
            raise NotImplementedError
        out, attn = super().forward(x,
                                    timesteps=timesteps,
                                    attn_mask=anything_mask,
                                    **kwargs)
        return out, attn


class SuperResModel(UNetModel):
    """A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode='bilinear')
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode='bilinear')
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
