import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List, Union


"""Attention modules.
"""
class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, 
                    query: torch.Tensor, 
                    key: torch.Tensor, 
                    value: torch.Tensor):
        """
        Args:
            query,key,value: shape (b, t, c)
        Returns:
            query,key,value: shape (b, nh, t, c//nh)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self,
                          value: torch.Tensor,
                          scores: torch.Tensor,
                          mask: torch.Tensor = None):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): shape: (b, nh, t2, c//nh).
            scores (torch.Tensor): shape: (b, nh, t1, t2).
            mask (torch.Tensor): attention padded mask, size (b, 1, t2) or (b, t1, t2)
        Returns:
            shape: (b, t1, c)
        """
        b = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = scores.softmax(dim=-1).masked_fill(mask, 0.0)
        else:
            attn = scores.softmax(dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)
        return self.linear_out(x)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True):
        """Multi-Head Attention layer with relative position encoding.
        Paper: https://arxiv.org/abs/1901.02860
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor = None,
        cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (torch.Tensor): shape (b, t1, c).
            key (torch.Tensor): shape (b, t2, c).
            value (torch.Tensor): shape (b, t2, c).
            mask (torch.Tensor): attention padded mask, shape (b, 1, t2) or (b, t1, t2).
            pos_emb (torch.Tensor): Positional embedding tensor (b, 2*t1-1, c).
            cache (torch.Tensor): Cache tensor (1, nh, cache_t, d_k * 2).
        Returns:
            torch.Tensor: Output tensor (b, t1, d_model).
            torch.Tensor: Cache tensor (1, nh, cache_t + t1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        if cache is not None and cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)    # (batch, 2*time1-1, head, d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # matrix_bd: (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache


class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float=0.0, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1: self.pe.size(1) // 2 + size,
        ]
        return pos_emb


"""Other modules.
"""
class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=self.stride, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs
    

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class LinearNoSubsampling(torch.nn.Module):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, 
                 idim: int, 
                 odim: int, 
                 dropout_rate: float,
                 pos_enc_class: torch.nn.Module
        ):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb


"""Encoder layer & encoder
"""
class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        self.ff_scale = 1.0
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        att_cache: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (b, t, c)
            mask: self-attention padded mask, shape (b, 1, t) or (b, t, t)
            pos_emb: relative positional embedding, shape (b, t, 2t-1)
            att_cache: shape (1, nh, cache_t, d_k * 2)
        """
        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        # att_cache: (b, head, cache_t, d_k*2)
        x_att, new_att_cache = self.self_attn(x, x, x, pos_emb, mask, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x_ffn = self.feed_forward(x)
        x = residual + self.ff_scale * self.dropout(x_ffn)
        if not self.normalize_before:
            x = self.norm_ff(x)

        return x, new_att_cache


class UpsampleConformerEncoder(torch.nn.Module):

    def __init__(
        self,
        # Common
        input_size: int = 512,
        output_size: int = 512,
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        normalize_before: bool = True,
        # Input & upsampling
        up_stride: int = 2,
        pre_lookahead_len: int = 3,
        # Attention
        attention_heads: int = 4,
        key_bias: bool = True,
        # MLP
        linear_units: int = 2048,
        # Dropouts
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.up_stride = up_stride
        # Input embedding
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            # Positional encoding
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        # Look ahead
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=pre_lookahead_len)
        # Norm
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        # Act
        activation = torch.nn.SiLU()
        # Self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # Feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # 1st Conformer
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                # Self-attn
                RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                # FFN
                PositionwiseFeedForward(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])
        # Upsample 
        self.up_layer = Upsample1D(channels=output_size, out_channels=output_size, stride=up_stride)
        # Input embedding2
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            # Positional encoding
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        # 2nd Conformer
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                # Self-attn
                RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                # FFN
                PositionwiseFeedForward(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
            ) for _ in range(num_up_blocks)
        ])

    """For non-streaming inference.
    """
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        # attention mask BEFORE upsample
        attn_mask1: torch.Tensor=None,
        # attention mask AFTER upsample
        attn_mask2: torch.Tensor=None,
    ) -> torch.Tensor:
        """
        Args:
            xs: shape (b, t, c)
            xs_lens: shape (b,)
            attn_mask1: (token level) shape (b, t, t)
            attn_mask2: (mel level) shape (b, 2t, 2t)
        """
        # Input & lookahead
        xs, pos_emb = self.embed(xs)
        xs = self.pre_lookahead_layer(xs)
        
        # 1st Conformer
        for block in self.encoders:
            xs, _ = block(xs, mask=attn_mask1, pos_emb=pos_emb)
        
        # Upsample to mel-level
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        # Input
        xs, pos_emb = self.up_embed(xs)
        
        # 2nd Conformer
        for block in self.up_encoders:
            xs, _ = block(xs, mask=attn_mask2, pos_emb=pos_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs
