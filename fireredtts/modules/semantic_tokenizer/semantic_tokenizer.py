from functools import reduce
from tokenize import Triple
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.utils.checkpoint import checkpoint

import einops
import math
import numpy as np
import os
import random
import torch
import torchaudio
import typing as tp
import warnings

from .audio import TorchMelSpectrogram
from .ecapa_tdnn import ECAPA_TDNN
from .hubert import HuBERT
from ..acoustic_codec.vector_quantization import VectorQuantization


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)
NORM = "weight_norm"


def get_mask_from_lengths(lengths, max_len=None):
    max_len = torch.max(lengths).item() if max_len is None else max_len
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = ~(ids < lengths.unsqueeze(1)).bool()
    return mask


class ConvLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "weight_norm",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class SConvTranspose1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "weight_norm",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SLSTM(nn.Module):

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        if bidirectional:
            self.lstm = nn.LSTM(
                dimension, dimension // 2, num_layers, bidirectional=bidirectional
            )
        else:
            self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()

        self.layers = nn.Sequential(
            SConv1d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                groups=groups,
                norm=NORM,
            ),
            Swish(),
            SConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                norm=NORM,
            ),
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            Swish(),
            SConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                norm=NORM,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        out_channels = in_channels
        self.layers = nn.Sequential(
            SConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                norm=NORM,
            ),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, strides=[2, 2], checkpointing=True):
        super().__init__()
        self.checkpointing = checkpointing

        self.downsample_scale = np.cumprod(np.asarray(strides))[-1]
        self.layers = [
            SConv1d(in_channels=C, out_channels=D, kernel_size=3, norm=NORM),
            Swish(),
        ]
        for stride in strides:
            self.layers += [
                EncoderBlock(out_channels=D, stride=stride),
                Swish(),
            ]
        self.layers += [
            SConv1d(in_channels=D, out_channels=D, kernel_size=3, norm=NORM),
            SLSTM(D, num_layers=1, bidirectional=True),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.checkpointing:
            x = checkpoint(
                self.layers, x.transpose(1, 2), use_reentrant=False
            ).transpose(1, 2)
        else:
            x = self.layers(x.transpose(1, 2)).transpose(1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, C, D, H, strides=[2, 2], checkpointing=True):
        super().__init__()
        self.checkpointing = checkpointing

        self.in_layer = nn.Sequential(
            SConv1d(in_channels=D, out_channels=H, kernel_size=3, norm=NORM),
            SLSTM(H, num_layers=1, bidirectional=True),
        )
        self.layers = nn.ModuleList()
        for stride in strides:
            self.layers.append(
                nn.Sequential(DecoderBlock(in_channels=H, stride=stride), Swish())
            )
        self.out_layer = SConv1d(
            in_channels=H, out_channels=C, kernel_size=3, norm=NORM
        )

    def forward(self, x, g=None):
        if self.checkpointing:
            y = checkpoint(self._forward, x, g, use_reentrant=False)
        else:
            y = self._forward(x, g)
        return y

    def _forward(self, x, g=None):
        h = self.in_layer(x.transpose(1, 2))

        for layer in self.layers:
            up_g = g.unsqueeze(-1).repeat(1, 1, h.shape[-1])
            h = h + up_g
            h = layer(h)

        y = self.out_layer(h)

        return y.transpose(1, 2), h.transpose(1, 2)


class TimeRegulator(nn.Module):

    def __init__(self, in_dim, scale, learnable=False):
        super().__init__()
        self.scale = scale
        self.learnable = learnable

    def forward(self, x, x_len, downsample=True):
        if downsample:
            x = self.downsample(x, x_len)
        else:
            x = self.upsample(x, x_len)
        return x

    def downsample(self, x, x_len):
        x = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), self.scale, stride=self.scale, ceil_mode=True
        ).transpose(1, 2)
        x_len = (x_len / self.scale).ceil()
        return x, x_len

    def upsample(self, x, x_len):
        if self.learnable:
            x = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = torch.repeat_interleave(x, self.scale, dim=1)
        return x


class TreeVectorQuantization(nn.Module):

    def __init__(
        self,
        in_dim,
        vq_class="VectorQuantization",
        vq_config={},
        tree_config={},
    ):
        super().__init__()
        self.vq_config = vq_config
        self.tree_config = tree_config

        self.quantizers = nn.ModuleList()
        self.time_regulators = nn.ModuleList()
        for config in self.tree_config:
            vq_config = self.vq_config.copy()
            if not isinstance(vq_config["codebook_size"], (tuple, list)):
                vq_config["codebook_size"] = [vq_config["codebook_size"]]
                vq_config["codebook_dim"] = [vq_config["codebook_dim"]]
            vq_config["codebook_size"] = vq_config["codebook_size"] * config["n_groups"]
            vq_config["codebook_dim"] = vq_config["codebook_dim"] * config["n_groups"]
            self.quantizers.append(
                VectorQuantization(
                    in_dim,
                    n_groups=config.get("n_groups", 1),
                    dropout_rate_per_group=config.get("dropout_rate_per_group", 0),
                    ordered=config.get("ordered", False),
                    **vq_config,
                )
            )
            self.time_regulators.append(
                TimeRegulator(
                    in_dim,
                    config["downsample_rate"],
                    config.get("learnable_time_regulator", False),
                )
            )

    def forward(
        self, inp, inp_len, enable_vq=True, update_codebook=True, return_pre_quant=False
    ):
        output, (quants, losses, embed_inds) = self.quantize(
            inp,
            inp_len,
            enable_vq=enable_vq,
            update_codebook=update_codebook,
            return_pre_quant=return_pre_quant,
        )
        loss = sum(losses) / len(losses)
        return output, (quants, loss, embed_inds)

    def quantize(
        self, inp, inp_len, enable_vq=True, update_codebook=True, return_pre_quant=False
    ):
        quants, losses, embed_inds = [], [], []

        pre_quant_output, quant_output, residual = 0, 0, inp
        for tree_config, quantizer, regulator in zip(
            self.tree_config, self.quantizers, self.time_regulators
        ):
            # Downsample
            x, x_len = regulator(residual, inp_len, True)

            # Quantization
            q, diff, embed_ind = quantizer(
                x,
                x_len,
                enable_vq=enable_vq,
                update_codebook=update_codebook,
                return_pre_quant=return_pre_quant,
            )
            if return_pre_quant:
                pq, q = q

            # Upsample
            x = regulator(q, x_len, False)[:, : residual.shape[1]]

            residual = residual - x
            quant_output = quant_output + x

            if return_pre_quant:
                pq = regulator(pq, x_len, False)[:, : residual.shape[1]]
                pre_quant_output = pre_quant_output + pq

            quants.append(q)
            losses.append(diff)
            embed_inds.append(embed_ind)

        if return_pre_quant:
            return (pre_quant_output, quant_output), (quants, losses, embed_inds)
        return quant_output, (quants, losses, embed_inds)

    def decode(self, seqs, seq_lens=None):
        if not isinstance(seqs, (tuple, list)):
            tokens, token_lens = self.deserialize(seqs, seq_lens)
        else:
            tokens, token_lens = seqs, seq_lens

        quant_output = 0
        for token, quantizer, regulator in zip(
            tokens, self.quantizers, self.time_regulators
        ):
            x = quantizer.decode(token).transpose(1, 2)
            x = regulator(x, None, False)
            if torch.is_tensor(quant_output):
                x = x[:, : quant_output.size(1)]
            quant_output = quant_output + x

        return quant_output, token_lens

    def serialize(self, tokens, token_lens):
        assert len(tokens) <= 2, "we only support 1 or 2-scale sequences now..."

        scale = self.tree_config[0]["downsample_rate"]
        token_lens = ((token_lens.float() / scale).ceil() * scale).int()

        seq1 = tokens[0].unsqueeze(-1)

        if len(tokens) == 1:
            seq_cat = seq1.view(seq1.shape[0], -1)
            seq_cat_lens = (token_lens / scale * seq1.shape[2]).int()
        elif len(tokens) == 2:
            seq2 = F.pad(
                tokens[1], (0, token_lens.max() - tokens[1].size(1)), "replicate"
            )
            seq2 = torch.stack([seq2[:, i::scale] for i in range(scale)], dim=-1)
            seq_cat = torch.cat((seq1, seq2), dim=-1).view(seq1.shape[0], -1)
            seq_cat_lens = (token_lens / scale + token_lens).int()

        return seq_cat, seq_cat_lens

    def deserialize(self, seqs, seq_lens):
        if len(self.tree_config) == 1:
            return [seqs], seq_lens

        max_scale = max(config["downsample_rate"] for config in self.tree_config)
        total_scale = sum(config["downsample_rate"] for config in self.tree_config)

        # Cut for aligning
        if seq_lens is None:
            seq_lens = torch.full([seqs.shape[0]], seqs.shape[1]).to(seqs.device)
        seq_lens = (seq_lens / total_scale).int() * total_scale
        token_lens = (seq_lens / total_scale).int() * max_scale
        seqs = seqs[:, : seq_lens.max()]

        # Separate
        tokens = torch.stack(
            [seqs[:, i::total_scale] for i in range(total_scale)], dim=-1
        )
        seq1 = tokens[..., 0]
        seq2 = tokens[..., 1:].contiguous().view(tokens.shape[0], -1)

        return [seq1, seq2], token_lens


class SemanticVQVAE(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        n_model_size,
        downsample_scales=[1, 2],
        upsample_scales=[[2, 1], [2, 1]],
        mel_config={},
        ssl_config={},
        # Quantization
        vq_class="VectorQuantization",
        vq_config={},
        tree_config={},
        # Training
        checkpointing=True,
        dual_decoding=False,
        n_samples_per_token=640,
        online_extraction=True,
        ssl_extractor=None,
    ):
        super(SemanticVQVAE, self).__init__()
        self.in_dim = in_dim
        self.n_model_size = n_model_size
        self.mel_config = mel_config
        self.dual_decoding = dual_decoding
        self.vq_config = vq_config
        self.tree_config = tree_config
        self.output_feature = "mel"
        self.n_samples_per_token = n_samples_per_token
        self.checkpointing = checkpointing

        self.mel_spectrogram = TorchMelSpectrogram(**mel_config)

        # Speaker encoder
        self.speaker_encoder = ECAPA_TDNN(
            out_dim,
            n_model_size,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=4,
            se_channels=128,
            global_context=True,
            batch_norm=True,
        )

        # Encoder & decoder
        self.encoder = Encoder(
            in_dim, n_model_size, downsample_scales, checkpointing=checkpointing
        )

        # Quantization
        self.quantizer = TreeVectorQuantization(
            n_model_size,
            vq_class=vq_class,
            vq_config=vq_config,
            tree_config=tree_config,
        )

    def forward(
        self,
        wav,
        wav_length,
        enable_vq=True,
        decode=True,
        extract_spk=True,
        shuffle=False,
        **kwargs,
    ):
        output_dict = {}

        with torch.no_grad():
            # Pad waveform
            if wav.shape[1] % self.n_samples_per_token > 0:
                pad_size = (
                    self.n_samples_per_token - wav.shape[1] % self.n_samples_per_token
                )
                wav = F.pad(wav, (0, pad_size), value=0)
                wav_length += pad_size

            # Extract mel & sll
            mel, mel_length = kwargs.get("mel", None), kwargs.get("mel_length", None)
            if mel is None:
                mel, mel_length = self.mel_spectrogram(wav, wav_length)
            output_dict.update({"mel": mel, "mel_length": mel_length})

            ssl, ssl_length = kwargs.get("ssl", None), kwargs.get("ssl_length", None)
            if ssl is None:
                ssl, ssl_length = self.ssl_extractor(wav, wav_length)
            output_dict.update({"ssl": ssl.float(), "ssl_length": ssl_length})

        input, input_length = ssl, ssl_length
        output, output_length = mel, mel_length

        encoder_outputs = self.encoder(input)
        quant_length = torch.ceil(input_length / self.encoder.downsample_scale)
        quant_length = quant_length.clamp(max=encoder_outputs.shape[1])

        quant, (quants, diff, embed_ind) = self.quantizer(
            encoder_outputs,
            quant_length,
            enable_vq=enable_vq,
            update_codebook=True,
            return_pre_quant=self.dual_decoding,
        )

        output_dict.update(
            {
                "quants": quants,
                "token": embed_ind,
                "token_length": quant_length.int(),
                "encoder_diffs": diff,
            }
        )

        # Speaker
        if extract_spk:
            cond, cond_length = output, output_length
            speaker_embedding = self.speaker_encoder(cond, cond_length)
            speaker_embedding_1 = speaker_embedding_2 = speaker_embedding
            output_dict["spk"] = speaker_embedding

        return output_dict

    @torch.no_grad
    def extract_speech_tokens(
        self, wav, wav_length, serialize=True, extract_spk=True, shuffle=False
    ):
        output_dict = self.forward(
            wav, wav_length, True, False, extract_spk=extract_spk, shuffle=shuffle
        )
        token_seqs, token_length = output_dict["token"], output_dict["token_length"]

        # Align sequences
        scale = self.tree_config[0]["downsample_rate"]
        token_length = (torch.ceil(token_length / scale) * scale).int()

        new_token_seqs, new_token_lens = [], []
        for i, token_seq in enumerate(token_seqs):
            # discrete-continuous tokens
            residual = None
            if isinstance(token_seq, (tuple, list)):
                token_seq, residual = token_seq

            scale = self.tree_config[i]["downsample_rate"]
            new_token_len = token_length // scale
            pad = int(new_token_len.max()) - token_seq.shape[1]
            token_seq = F.pad(
                token_seq,
                (0, pad) if len(token_seq.shape) == 2 else (0, 0, 0, pad),
                "replicate",
            )

            if residual is not None:
                token_seq = (token_seq, residual)
            new_token_seqs.append(token_seq)
            new_token_lens.append(new_token_len)

        if len(new_token_seqs) == 1:
            new_token_seqs, new_token_lens = new_token_seqs[0], new_token_lens[0]
        elif serialize:
            new_token_seqs, new_token_lens = self.quantizer.serialize(
                new_token_seqs, new_token_lens
            )

        output_dict.update(
            {
                "embed": output_dict["quants"],
                "token": new_token_seqs,
                "token_length": new_token_lens,
            }
        )

        return output_dict

    @torch.no_grad
    def code_to_latent(self, token, mel=None):
        quant, _ = self.quantizer.decode(token, None)
        speaker_embedding = self.speaker_encoder(mel)
        latents = quant + speaker_embedding.unsqueeze(1).repeat(1, quant.shape[1], 1)
        return {
            "latents": latents,
        }
