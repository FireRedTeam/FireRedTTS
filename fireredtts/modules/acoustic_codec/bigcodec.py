from einops import rearrange
from torch import sin, pow
from torch.nn import Parameter
from torch.nn.utils import spectral_norm, weight_norm

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import typing as tp
import warnings

from .alias_free_torch import *
from .vector_quantization import VectorQuantization


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


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "time_group_norm":
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
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
        **kwargs,
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
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
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


def WNConv1d(*args, **kwargs):
    if kwargs.get("causal", False):
        kwargs["norm"] = "weight_norm"
        conv1d = SConv1d(*args, **kwargs)
    else:
        kwargs.pop("causal")
        conv1d = weight_norm(nn.Conv1d(*args, **kwargs))
    return conv1d


def WNConvTranspose1d(*args, **kwargs):
    if kwargs.get("causal", False):
        kwargs["norm"] = "weight_norm"
        transposed_conv1d = SConvTranspose1d(*args, **kwargs)
    else:
        kwargs.pop("causal")
        transposed_conv1d = weight_norm(nn.ConvTranspose1d(*args, **kwargs))
    return transposed_conv1d


class SnakeBeta(nn.Module):

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class ResidualUnit(nn.Module):

    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True), causal=causal),
            WNConv1d(
                dim, dim, kernel_size=7, dilation=dilation, padding=pad, causal=causal
            ),
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True), causal=causal),
            WNConv1d(dim, dim, kernel_size=1, causal=causal),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):

    def __init__(
        self, dim: int = 16, stride: int = 1, dilations=(1, 3, 9), causal: bool = False
    ):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d, causal=causal) for d in dilations]
        self.block = nn.Sequential(
            *runits,
            Activation1d(
                activation=SnakeBeta(dim // 2, alpha_logscale=True), causal=causal
            ),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                causal=causal,
            ),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        dilations=(1, 3, 9),
        causal: bool = False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(
                activation=SnakeBeta(input_dim, alpha_logscale=True), causal=causal
            ),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2,
                causal=causal,
            ),
        )
        self.block.extend(
            [ResidualUnit(output_dim, dilation=d, causal=causal) for d in dilations]
        )

    def forward(self, x):
        return self.block(x)


class ResLSTM(nn.Module):

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True,
    ):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension,
            dimension if not bidirectional else dimension // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y


class Resampler(nn.Module):

    def __init__(self, source_sr=24000, target_sr=24000):
        super().__init__()
        self.source_sr = source_sr
        self.target_sr = target_sr

    def forward(self, wav, wav_length):
        if self.source_sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, self.source_sr, self.target_sr)
            wav_length = (wav_length * (self.source_sr / self.target_sr)).int()
        return wav, wav_length


class CodecEncoder(nn.Module):

    def __init__(
        self,
        ngf=48,
        use_rnn=True,
        rnn_bidirectional=False,
        rnn_num_layers=2,
        up_ratios=(2, 2, 2, 5, 5),
        dilations=(1, 3, 9),
        out_channels=1024,
        causal=False,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3, causal=causal)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [
                EncoderBlock(d_model, stride=stride, dilations=dilations, causal=causal)
            ]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(
                    d_model, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional
                )
            ]
        # Create last convolution
        self.block += [
            Activation1d(
                activation=SnakeBeta(d_model, alpha_logscale=True), causal=causal
            ),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1, causal=causal),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class CodecDecoder(nn.Module):

    def __init__(
        self,
        in_channels=1024,
        upsample_initial_channel=1536,
        ngf=48,
        use_rnn=True,
        rnn_bidirectional=False,
        rnn_num_layers=2,
        up_ratios=(5, 5, 2, 2, 2),
        dilations=(1, 3, 9),
        causal=False,
        delay=0,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.delay = delay

        channels = upsample_initial_channel
        layers = [
            WNConv1d(in_channels, channels, kernel_size=7, padding=3, causal=causal)
        ]

        if use_rnn:
            layers += [
                ResLSTM(
                    channels, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional
                )
            ]

        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(input_dim, output_dim, stride, dilations, causal=causal)
            ]

        layers += [
            Activation1d(
                activation=SnakeBeta(output_dim, alpha_logscale=True), causal=causal
            ),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3, causal=causal),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, x):
        # Time delay
        if self.delay > 0:
            x = F.pad(x, (0, self.delay), mode="constant", value=0)

        x = self.model(x)

        # De-delay
        if self.delay > 0:
            x = x[..., self.delay :]

        return x

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class BigCodec(nn.Module):

    def __init__(
        self,
        n_model_size: int,
        encoder_config: dict,
        decoder_config: dict,
        vq_config: dict,
        resampler_config: dict = None,
    ):
        super(BigCodec, self).__init__()
        self.n_model_size = n_model_size

        self.encoder = CodecEncoder(out_channels=n_model_size, **encoder_config)
        self.decoder = CodecDecoder(in_channels=n_model_size, **decoder_config)
        self.quantizer = VectorQuantization(n_model_size, **vq_config)

        # Optional modules
        if resampler_config:
            self.resampler = Resampler(**resampler_config)

    def forward(
        self, wav, wav_length=None, enable_vq=True, decode=True, update_codebook=True
    ):
        # Preprocess wav
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(1)
        if wav_length is None:
            wav_length = torch.full([wav.shape[0]], max(wav.shape)).to(wav.device)

        # (Optional) Resample
        processed_wav, processed_wav_length = wav, wav_length
        if hasattr(self, "resampler"):
            processed_wav, processed_wav_length = self.resampler(
                processed_wav, processed_wav_length
            )

        # Update VQ parameters
        quant_length = torch.ceil(processed_wav_length / self.encoder.hop_length).int()
        update_codebook = update_codebook and self.training

        # Encode
        encoder_outputs = self.encoder(processed_wav)

        # Quantize
        quant, diff, embed_ind = self.quantizer(
            encoder_outputs.transpose(1, 2),
            quant_length.clamp(max=encoder_outputs.shape[2]),
            enable_vq=enable_vq,
            update_codebook=update_codebook,
        )

        if decode:
            # Decode
            decoder_outputs = self.decoder(quant.transpose(1, 2))
        else:
            decoder_outputs = None

        output_dict = {
            "quant": quant,
            "token": embed_ind,
            "token_length": quant_length,
            "encoder_diffs": diff,
            "wav_pred": decoder_outputs,
        }
        return output_dict

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def extract_speech_tokens(
        self, wav, wav_length, serialize=True, extract_spk=True, shuffle=False
    ):
        output_dict = self.forward(wav, wav_length, enable_vq=True, decode=False)
        token_seqs, token_length = [output_dict["token"]], [output_dict["token_length"]]
        output_dict.update(
            {
                "token": token_seqs,
                "token_length": token_length,
            }
        )

        return output_dict

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def reconstruct_wav(self, token=None, quant=None, spk=None):
        if token is not None:
            # De-tokenization
            quant = self.quantizer.decode(token)

            # Speaker embedding
            if hasattr(self, "global_encoder"):
                quant = quant + spk.unsqueeze(2)
        else:
            assert quant is not None

        # Decode
        wav_pred = self.decoder(quant)

        return {
            "wav_pred": wav_pred,
        }
