from einops import rearrange, repeat
from torch import nn
from typing import Union

import torch
import torch.nn.functional as F
import typing as tp
import numpy as np
import warnings


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def flatten(x, x_len):
    x_f = x.view(-1, *x.shape[2:])
    return x_f


def ema_inplace(moving_avg, new, decay):
    if isinstance(decay, torch.Tensor):
        moving_avg.data.mul_(decay).add_(new * (1 - decay))
    else:
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


class EuclideanCodebook(nn.Module):

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 1.0,
        n_cache_iters: int = 1,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.update_iter = 0

        self.n_cache_iters = n_cache_iters
        self.cache_vectors = []
        self.cache_indices = []

        if isinstance(self.decay, (tuple, list)):
            self.embed_avg_cache = []
            self.register_buffer("diff_avg_long", torch.zeros(codebook_size) + 1e-5)
            self.register_buffer("diff_avg_short", torch.zeros(codebook_size) + 1e-5)
        self.register_buffer("inited", torch.Tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

    def replace_(self, samples, mask, dists=None):
        reset_cluster_size = min(
            self.threshold_ema_dead_code + 1, self.threshold_ema_dead_code * 1.1
        )

        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        modified_codebook_avg = torch.where(
            mask[..., None], modified_codebook * reset_cluster_size, self.embed_avg
        )
        modified_cluster_size = torch.where(
            mask,
            torch.full_like(self.cluster_size, reset_cluster_size),
            self.cluster_size,
        )

        self.embed.data.copy_(modified_codebook)
        self.embed_avg.data.copy_(modified_codebook_avg)
        self.cluster_size.data.copy_(modified_cluster_size)

    def expire_codes_(self, batch_samples, dists=None):
        self.update_iter += 1
        if self.threshold_ema_dead_code == 0:
            return
        elif self.threshold_ema_dead_code < 1:
            threshold_ema_dead_code = (
                sum(self.cluster_size) * self.threshold_ema_dead_code
            )
        else:
            threshold_ema_dead_code = self.threshold_ema_dead_code

        expired_codes = self.cluster_size < threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes, dists=dists)

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices

        return embed_ind, dist

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind, dist = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)

        return embed_ind, dist

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x, x_len, enable_vq=True, update_codebook=True, masking=False):
        x_org, shape, dtype = x, x.shape, x.dtype

        x = self.preprocess(x)

        embed_ind, dist = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        dist = dist.view(shape[0], shape[1], dist.shape[-1])

        quantize = self.dequantize(embed_ind)

        if self.training and update_codebook:
            if enable_vq:
                quantize = x_org + (quantize - x_org).detach()
            else:
                quantize = x_org

            # Get flatten embedding indices and distances
            if masking:
                x_f = torch.cat(
                    [e[: int(e_len)] for e, e_len in zip(x_org, x_len)], dim=0
                )
                embed_ind_f = torch.cat(
                    [e[: int(e_len)] for e, e_len in zip(embed_ind, x_len)], dim=0
                )
                dist_f = torch.cat(
                    [e[: int(e_len)] for e, e_len in zip(dist, x_len)], dim=0
                )
                q_f = torch.cat(
                    [e[: int(e_len)] for e, e_len in zip(quantize.detach(), x_len)],
                    dim=0,
                )
                commit_loss = F.mse_loss(q_f, x_f)
            else:
                x_f = x_org.view(-1, x_org.shape[-1]).contiguous()
                embed_ind_f = embed_ind.view(-1).contiguous()
                dist_f = dist.view(-1).contiguous()
                commit_loss = F.mse_loss(quantize.detach(), x_org)
            self.init_embed_(x_f)

            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x_f, dist_f)

            # Calculate codebook statistics
            embed_onehot = F.one_hot(embed_ind_f, self.codebook_size).type(dtype)
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x_f.t() @ embed_onehot

            # EMA updating
            ema_inplace(self.cluster_size, embed_onehot_sum, self.decay)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)

            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        else:
            commit_loss = torch.tensor(
                0.0, device=quantize.device, requires_grad=self.training
            )

        return quantize, commit_loss, embed_ind


class MultiHeadEuclideanCodebook(nn.Module):

    def __init__(
        self,
        dim: Union[int, list],
        codebook_size: list,
        n_groups: int = 1,
        dropout_rate_per_group: float = 0,
        ordered: bool = False,
        ordered_axis: str = "sequence",
        method: str = "product",
        **kwargs,
    ):
        super().__init__()
        self.codebook_sizes = codebook_size
        self.codebook_dims = dim
        self.n_groups = n_groups
        self.n_heads_per_group = len(codebook_size) // n_groups
        self.dropout_rate_per_group = dropout_rate_per_group
        self.ordered = ordered
        self.ordered_axis = ordered_axis
        self.method = method
        assert len(codebook_size) % n_groups == 0

        self.codebooks = nn.ModuleList()
        dim = self.codebook_dims
        for i, size in enumerate(self.codebook_sizes):
            if isinstance(self.codebook_dims, list):
                dim = (
                    self.codebook_dims[i]
                    if method == "product"
                    else sum(self.codebook_dims)
                )
            self.codebooks.append(EuclideanCodebook(dim, size, **kwargs))

    def decode(self, embed_ind):
        if self.n_groups == 1 or len(embed_ind.shape) == 2:
            embed_ind = embed_ind.unsqueeze(-1)

        actual_n_groups = embed_ind.shape[-1]
        if actual_n_groups < self.n_groups:
            print(
                f"The actual number of heads ({actual_n_groups}) is smaller than the pre-designed ({self.n_groups})!"
            )
            embed_ind = F.pad(
                embed_ind, (0, self.n_groups - actual_n_groups), "replicate"
            )
        # assert embed_ind.shape[-1] == self.n_groups

        index_heads, codebook_heads, scale_heads = zip(
            *[
                (
                    embed_ind[..., i // self.n_heads_per_group],
                    self.codebooks[i : i + self.n_heads_per_group],
                    self.codebook_sizes[i : i + self.n_heads_per_group],
                )
                for i in range(0, len(self.codebook_sizes), self.n_heads_per_group)
            ]
        )

        quantize_heads, quantize_groups = [], []
        for i in range(self.n_groups):
            embed_ind, codebooks, scales = (
                index_heads[i],
                codebook_heads[i],
                scale_heads[i],
            )

            inv_scales = list(torch.tensor([1] + scales[:-1]).cumprod(dim=0))[::-1]
            inv_quantizes = []
            for codebook, scale in zip(codebooks[::-1], inv_scales):
                index, embed_ind = embed_ind // scale, embed_ind % scale
                quantize = codebook.dequantize(index)
                inv_quantizes.append(quantize)
            quantizes = inv_quantizes[::-1]
            group_embeddings = torch.cat(quantizes, dim=-1)
            quantize_groups.append(group_embeddings)
            quantize_heads += quantizes

        if self.method == "product":
            if actual_n_groups < self.n_groups:
                for i in range(actual_n_groups, self.n_groups):
                    quantize_groups[i].zero_()
            quantize = torch.cat(quantize_groups, dim=-1)
        elif self.method == "residual":
            quantize = sum(quantize_heads)

        return quantize

    def forward(self, x, x_len, enable_vq=True, update_codebook=True):
        # Pre-process
        x = self._preprocess(x)

        # Quantize
        quants, losses, indices = self._quantize(
            x, x_len, enable_vq=enable_vq, update_codebook=update_codebook
        )

        # Integrate
        quant, loss, index = self._integrate(
            quants, losses, indices, update_codebook=update_codebook
        )

        return quant, loss, index

    def _preprocess(self, x):
        if self.method == "product" and isinstance(self.codebook_dims, (list, tuple)):
            x = torch.split(x, self.codebook_dims, dim=-1)
        return x

    def _quantize(self, x, x_len, enable_vq, update_codebook):
        if self.method == "product":
            quants, losses, indices = zip(
                *[
                    codebook(
                        chunk,
                        x_len,
                        enable_vq=enable_vq,
                        update_codebook=update_codebook,
                    )
                    for chunk, codebook in zip(x, self.codebooks)
                ]
            )
        elif self.method == "residual":
            quants, losses, indices = [], [], []
            residual = x
            for codebook in self.codebooks:
                quant, loss, index = codebook(
                    residual,
                    x_len,
                    enable_vq=enable_vq,
                    update_codebook=update_codebook,
                )
                residual = residual - quant
                quants.append(quant)
                losses.append(loss)
                indices.append(index)

        return quants, losses, indices

    def _integrate(self, quants, losses, indices, update_codebook=True):
        (B, T, D), M = quants[0].shape, len(quants)
        device = quants[0].device

        # Average loss
        loss = sum(losses) / len(losses)

        # Get indices
        if self.n_groups == 1:
            scale = (
                torch.tensor([1] + self.codebook_sizes[:-1]).cumprod(dim=0).to(device)
            )
            index = (torch.stack(indices, dim=-1) * scale).sum(dim=-1)
        else:
            index_heads, scale_heads = zip(
                *[
                    (
                        indices[i : i + self.n_heads_per_group],
                        torch.tensor(
                            [1]
                            + self.codebook_sizes[i : i + self.n_heads_per_group - 1]
                        )
                        .cumprod(dim=0)
                        .to(device),
                    )
                    for i in range(0, len(quants), self.n_heads_per_group)
                ]
            )
            index = torch.stack(
                [
                    (torch.stack(x, dim=-1) * s).sum(dim=-1)
                    for x, s in zip(index_heads, scale_heads)
                ],
                dim=-1,
            )

        # Add dropout
        quant_groups = self._dropout(quants, enabled=update_codebook)

        # Combine quantized features
        if self.method == "product":
            quant = torch.cat(quant_groups, dim=-1)
        elif self.method == "residual":
            quant = torch.cat(quant_groups, dim=-1).view(B, T, M, D).sum(dim=2)

        return quant, loss, index

    def _dropout(self, quants, enabled=True):
        if enabled and self.training and self.ordered:
            if self.dropout_rate_per_group == 0:
                threshold = [
                    (i // self.n_heads_per_group * 1.0 / self.n_groups)
                    for i in range(0, len(quants), self.n_heads_per_group)
                ]
            elif self.dropout_rate_per_group == "exp":
                x = [np.exp(4 * i / self.n_groups) for i in range(self.n_groups)]
                x = np.asarray(x) / sum(x)
                threshold = np.cumsum(np.asarray([0] + x))
            else:
                x = np.asarray(self.dropout_rate_per_group) / sum(
                    self.dropout_rate_per_group
                )
                threshold = np.cumsum(np.asarray([0] + x))

            if self.ordered_axis == "sequence":
                rate = torch.rand((quants[0].shape[0], 1, 1), device=quants[0].device)
            elif self.ordered_axis == "frame":
                rate = torch.rand(
                    (quants[0].shape[0], quants[0].shape[1], 1), device=quants[0].device
                )

            quant_groups = []
            for i in range(0, len(quants), self.n_heads_per_group):
                quant_group = torch.cat(quants[i : i + self.n_heads_per_group], dim=-1)
                is_kept = threshold[i // self.n_heads_per_group] <= rate
                quant_group = torch.where(
                    is_kept, quant_group, torch.zeros_like(quant_group)
                )
                quant_groups.append(quant_group)
        elif self.ordered:
            quant_groups = []
            for i in range(0, len(quants), self.n_heads_per_group):
                quant_group = torch.cat(quants[i : i + self.n_heads_per_group], dim=-1)
                quant_groups.append(quant_group)
        else:
            quant_groups = quants

        return quant_groups


class VectorQuantization(nn.Module):

    def __init__(
        self,
        dim: int,
        codebook_size: Union[int, list],
        codebook_dim: Union[int, list] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 1.0,
        commitment_weight: float = 1.0,
        requires_projection: bool = False,
        norm: str = "none",
        **kwargs,
    ):
        super().__init__()
        _codebook_dim: Union[int, list] = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim or requires_projection
        proj_dim = (
            sum(_codebook_dim) if isinstance(_codebook_dim, list) else _codebook_dim
        )
        if requires_projection:
            self.project_in = nn.Linear(dim, proj_dim)
            self.project_out = nn.Linear(proj_dim, dim)
            if norm == "weight_norm":
                self.project_in = torch.nn.utils.weight_norm(self.project_in)
                self.project_out = torch.nn.utils.weight_norm(self.project_out)
        else:
            self.norm = None
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self.codebook_size = codebook_size

        codebook_class = (
            EuclideanCodebook
            if isinstance(codebook_size, int)
            else MultiHeadEuclideanCodebook
        )
        self._codebook = codebook_class(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
            **kwargs,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x, x_len=None):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind, embed_len=None):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def decode_latent(self, latent, latent_len=None):
        if latent_len is None:
            latent_len = (
                torch.Tensor([latent.shape[1]] * latent.shape[0])
                .to(latent.device)
                .int()
            )

        quantize, _, _ = self._codebook(latent, latent_len)
        quantize = self.project_out(quantize)
        return quantize

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        x,
        x_len,
        enable_vq=True,
        update_codebook=True,
        return_pre_quant=False,
        return_dict=False,
    ):
        device = x.device

        x = self.project_in(x)

        quantize, commit_loss, embed_ind = self._codebook(
            x, x_len, enable_vq=enable_vq, update_codebook=update_codebook
        )
        if self.training and update_codebook:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            if self.commitment_weight > 0:
                loss = loss + commit_loss * self.commitment_weight
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=False)

        embed = quantize
        quantize = self.project_out(quantize)

        if return_dict:
            return {
                "quantize": quantize,
                "loss": loss,
                "embed": embed,
                "embed_ind": embed_ind,
            }
        elif return_pre_quant:
            pre_quantize = self.project_out(x)
            return (pre_quantize, quantize), loss, embed_ind
        else:
            return quantize, loss, embed_ind
