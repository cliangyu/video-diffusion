from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    frame_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, attn_mask, T=1, frame_indices=None, attn_weights_list=None):
        for layer in self:
            kwargs = {}
            if isinstance(layer, TimestepBlock):
                kwargs['emb'] = emb
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs['temb'] = emb
                kwargs['attn_mask'] = attn_mask
                kwargs['T'] = T
                kwargs['attn_weights_list'] = attn_weights_list
                kwargs['frame_indices'] = frame_indices
            x = layer(x, **kwargs)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

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
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

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
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

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
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

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


class AugmentedTransformer(nn.Module):
    valid_augment_types = [
        'none',
        'add_presoftmax', 'add_presoftmax_time',
        'add', 'add_time',
        'mult', 'mult_time',
        'add_manyhead_presoftmax', 'add_manyhead_presoftmax_time',
        'add_manyhead', 'add_manyhead_time',
        'mult_manyhead_presoftmax', 'mult_manyhead_presoftmax_time',
        'mult_manyhead', 'mult_manyhead_time',
    ]
    compare = [
        'add_manyhead_presoftmax_time',
        'add_presoftmax_time',
        'add_manyhead_presoftmax',
        'add_manyhead_time',
        'mult_manyhead_time',
    ]

    def __init__(self, channels, num_heads, time_embed_dim, use_checkpoint=False, augment_type='none'):
        assert augment_type in self.valid_augment_types
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        self.use_checkpoint = use_checkpoint
        self.augment_type = augment_type
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        # augmentation stuff
        self.manyhead = 'manyhead' in augment_type
        self.use_time = 'time' in augment_type
        self.presoftmax = 'presoftmax' in augment_type
        self.multiplicative = 'mult'  in augment_type
        if not self.multiplicative:
            assert 'add' in augment_type or augment_type == 'none'
        # print("self.manyhead, self.use_time, self.presoftmax, self.multiplicative")
        # print(self.manyhead, self.use_time, self.presoftmax, self.multiplicative)
        if augment_type == 'none':
            pass
        else:
            self.augment_layer1 = conv_nd(2, 3, channels, 1)
            if self.use_time:
                self.augment_layer2 = conv_nd(2, time_embed_dim, channels, 1)
            self.augment_layer3 = conv_nd(2, channels, channels if self.manyhead else num_heads, 1)

    def forward(self, x, temb, frame_indices, attn_mask=None, attn_weights_list=None):
        out, attn = checkpoint(self._forward, (x, temb, frame_indices, attn_mask), self.parameters(), self.use_checkpoint)
        if attn_weights_list is not None:
            B, D, C, T = x.shape
            attn_weights_list.append(attn.detach().view(B*D, -1, T, T).mean(dim=1).abs())
        return out

    def _forward(self, x, temb, frame_indices, attn_mask):
        """
        x - shape B x D x C x T - batch x D (e.g. spatial dimension along which temb, frame_indices and attn_mask don't vary)
        temb - B x C x T (althout Tth dimension should all be the same)
        frame_indices - B x T
        attn_mask - B x T
        """
        B, D, C, T = x.shape
        qkv = self.qkv(self.norm(x.view(B*D, C, T))).view(B, D, 3, C//self.num_heads, self.num_heads, T)
        q, k, v = (qkv[:, :, i] for i in range(3))
        _, _, channels_per_head, _, _ =  q.shape
        scale = 1 / math.sqrt(math.sqrt(channels_per_head))
        weight = th.einsum(
            "bdcht,bdchs->bdhts", q * scale, k * scale  # BxHxTxT
        )  # More stable with f16 than dividing afterwards

        if self.augment_type != 'none':
            # compute w_aug
            relative = frame_indices.view(B, 1, 1, T) - frame_indices.view(B, 1, T, 1)  # BxTxTx1
            relative = th.cat([
                relative.float().clamp(min=0), (-relative).float().clamp(min=0), (relative == 0).float()
            ], dim=1)
            relative = th.log(1+relative)
            emb = self.augment_layer1(relative)
            if self.use_time:
                emb = emb + self.augment_layer2(temb.view(B, -1, T, 1))   # B x C x T x T
            w_aug = self.augment_layer3(th.relu(emb))   # BxCxTxT or BxHxTxT
            if not self.manyhead:
                w_aug = w_aug.repeat(1, channels_per_head, 1, 1)  # BxHxTxT
            w_aug = w_aug.view(B, 1, C, T, T)

        def softmax(w, attn_mask):
            if attn_mask is not None:
                inf_mask = (1-attn_mask)
                inf_mask[inf_mask == 1] = th.inf
                w = w - inf_mask.view(B, 1, 1, 1, T)  # BxDxCxTxT
            return th.softmax(w.float(), dim=-1).type(w.dtype)

        if self.augment_type == 'none':
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            h = th.einsum("bdhts,bdchs->bdcht", weight, v)
            assert h.shape == (B, D, channels_per_head, self.num_heads, T)
            h = h.reshape(B, D, C, T)
        else:
            weight = weight.repeat(1, 1, channels_per_head, 1, 1)  # BxDxCxTxT
            if self.presoftmax:
                weight = weight*w_aug if self.multiplicative else weight+w_aug
                weight = softmax(weight, attn_mask)
            else:
                weight = softmax(weight, attn_mask)
                weight = weight*w_aug if self.multiplicative else weight+w_aug
                if attn_mask is not None:
                    weight = weight * attn_mask.view(B, 1, 1, 1, T)
            v = v.view(B, D, C, T)
            h = th.einsum('bdcts,bdcs->bdct', weight, v)

        assert h.shape == (B, D, C, T)
        h = self.proj_out(h.view(B*D, C, T)).view(B, D, C, T)
        return x.view(B, D, C, T) + h, weight


class FactorizedAttentionBlock(nn.Module):
    """
    Loosely based on CSDI's factorized attention for time-series data.
    https://openreview.net/pdf?id=VzuIzbRDrum
    """
    def __init__(self, channels, num_heads=1, time_embed_dim=None, use_checkpoint=False, temporal_augment_type='none'):
        super().__init__()
        self.spatial_attention = AugmentedTransformer(
            channels=channels, num_heads=num_heads, time_embed_dim=time_embed_dim,
            augment_type='none')
        self.temporal_attention = AugmentedTransformer(
            channels=channels, num_heads=num_heads, time_embed_dim=time_embed_dim,
            augment_type=temporal_augment_type)


    def forward(self, x, attn_mask, temb, T, attn_weights_list=None, frame_indices=None):
        BT, C, H, W = x.shape
        B = BT//T
        x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
        x = x.reshape(B, H*W, C, T)
        x = self.temporal_attention(x,
                                    temb.view(B, T, -1).permute(0, 2, 1),  # B x C x T
                                    frame_indices,  # B x T
                                    attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2), # B x T
                                    attn_weights_list=None if attn_weights_list is None else attn_weights_list['temporal'],)
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
        x = x.reshape(BT, 1, C, H*W)
        x = self.spatial_attention(x, temb=None, frame_indices=None,
                                   attn_weights_list=None if attn_weights_list is None else attn_weights_list['spatial'])
        x = x.reshape(BT, C, H, W)
        return x


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention.
#     """

#     def forward(self, qkv, attn_mask):
#         """
#         Apply QKV attention.

#         :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
#         :attn_mask: an [N x T] binary tensor denoting what may be attended to
#         :return: an [N x C x T] tensor after attention.
#         """
#         n, ch3, t = qkv.shape
#         ch = ch3 // 3
#         q, k, v = th.split(qkv, ch, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts", q * scale, k * scale
#         ).float()  # More stable with f16 than dividing afterwards
#         if attn_mask is not None:
#             inf_mask = (1-attn_mask)
#             inf_mask[inf_mask == 1] = th.inf
#             weight = weight - inf_mask.view(n, 1, t)
#         weight = th.softmax(weight, dim=-1).type(weight.dtype)
#         return th.einsum("bts,bcs->bct", weight, v), weight

#     @staticmethod
#     def count_flops(model, _x, y):
#         """
#         A counter for the `thop` package to count the operations in an
#         attention operation.

#         Meant to be used like:

#             macs, params = thop.profile(
#                 model,
#                 inputs=(inputs, timestamps),
#                 custom_ops={QKVAttention: QKVAttention.count_flops},
#             )

#         """
#         b, c, *spatial = y[0].shape
#         num_spatial = int(np.prod(spatial))
#         # We perform two matmuls with the same number of ops.
#         # The first computes the weight matrix, the second computes
#         # the combination of the value vectors.
#         matmul_ops = 2 * b * (num_spatial ** 2) * c
#         model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

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
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if ds in attention_resolutions and self.n_blocks_before_attn is None:
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
                        FactorizedAttentionBlock(ch, num_heads=num_heads, time_embed_dim=time_embed_dim,
                                                 use_checkpoint=use_checkpoint, temporal_augment_type=temporal_augment_type)
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        if self.n_blocks_before_attn is None:
            self.n_blocks_before_attn = len(self.input_blocks)
            first_attn_ds = ds
            first_attn_ch = ch

        if use_spatial_encoding:
            first_attn_res = image_size // first_attn_ds
            self.spatial_encoding = nn.Parameter(
                th.randn(1, first_attn_ch, first_attn_res, first_attn_res),
                requires_grad=True
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
            FactorizedAttentionBlock(ch, num_heads=num_heads, time_embed_dim=time_embed_dim,
                                     use_checkpoint=use_checkpoint, temporal_augment_type=temporal_augment_type),
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
                        ch + input_block_chans.pop(),
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
                        FactorizedAttentionBlock(ch, num_heads=num_heads, time_embed_dim=time_embed_dim,
                                                 use_checkpoint=use_checkpoint, temporal_augment_type=temporal_augment_type),
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None, attn_mask=None, T=1,
                return_attn_weights=False, frame_indices=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        attns = {'spatial': [], 'temporal': [], 'mixed': []} if return_attn_weights else None
        for layer, module in enumerate(self.input_blocks):   # add frame embedding after first input block
            h = module(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
            hs.append(h)
            if layer + 1 == self.n_blocks_before_attn:
                h = self.add_positional_encodings(h, frame_indices=frame_indices)
        h = self.middle_block(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        h = h.type(x.dtype)
        out = self.out(h)
        return out, attns

    def add_positional_encodings(self, h):
        if self.spatial_encoding is not None:
            h = h + self.spatial_encoding
        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

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
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class UNetVideoModel(UNetModel):

    def __init__(self, T,
                 use_frame_encoding,
                 cross_frame_attention,
                 enforce_position_invariance,
                 *args, **kwargs,
                 ):
        self.T = T
        self.use_frame_encoding = use_frame_encoding
        self.cross_frame_attention = cross_frame_attention
        self.enforce_position_invariance = enforce_position_invariance
        super().__init__(
            *args, **kwargs,
        )

    def forward(self, x, timesteps, frame_indices=None, **kwargs):
        B, T, C, H, W = x.shape
        if frame_indices is None:
            frame_indices = th.arange(0, T, device=x.device).view(1, T).expand(B, T)
        x = x.view(B*T, C, H, W)
        timesteps = timesteps.view(B, 1).expand(B, T).reshape(B*T)
        out, attn = super().forward(
            x, timesteps, frame_indices=frame_indices,
            T=T if self.cross_frame_attention else 1, **kwargs
        )
        return out.view(B, T, self.out_channels, H, W), attn

    def add_positional_encodings(self, h, frame_indices):
        h = super().add_positional_encodings(h)
        if not self.use_frame_encoding:
            return h
        B, T = frame_indices.shape
        BT, C, H, W = h.shape
        assert BT == B*T
        max_period = self.T*10
        if self.enforce_position_invariance:
            frame_indices = frame_indices.float() - frame_indices.float().mean(dim=1, keepdim=True)
        emb = frame_embedding(frame_indices, C, max_period=max_period)
        return h + emb.view(BT, C, 1, 1)


class CondMargVideoModel(UNetVideoModel):   # TODO could generalise to derive similar class for image model

    def __init__(self, **kwargs):
        kwargs['in_channels'] += 2
        super().__init__(**kwargs)

    def forward(self, x, x0, obs_mask, latent_mask, kinda_marg_mask, **kwargs):
        *leading_dims, C, H, W = x.shape
        indicator_template = th.ones_like(x[:, :, :1, :, :])
        obs_indicator = indicator_template * obs_mask
        kinda_marg_indicator = indicator_template * kinda_marg_mask
        x = th.cat([x*latent_mask + x0*obs_mask,
                    obs_indicator,
                    kinda_marg_indicator],
                   dim=2)
        attn_mask = (obs_mask + latent_mask + kinda_marg_mask).clip(max=1)
        out, attn = super().forward(x, attn_mask=attn_mask, **kwargs)
        return out*latent_mask, attn


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
