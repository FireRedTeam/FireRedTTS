import torch
import torch.nn.functional as F
from typing import Dict, List
from einops import pack, repeat
from fireredtts.modules.flow_v2.embedding import DualEmbedding
from fireredtts.modules.flow_v2.upsample_encoder import UpsampleConformerEncoder
from fireredtts.modules.flow_v2.estimator_dit import DiT


class CausalFmWithSpkCtx(torch.nn.Module):
    def __init__(
        self,
        # Basic in-out
        spk_channels: int,
        spk_enc_channels: int,  # out channels of spk & encoder projection
        # Module
        token_emb: DualEmbedding,
        encoder: UpsampleConformerEncoder,
        estimator: DiT,
        # Flow cfg
        infer_cfg_rate: float = 0.7,
    ):
        super().__init__()
        # Variants
        self.up_stride = encoder.up_stride
        self.infer_cfg_rate = infer_cfg_rate
        # Module
        self.spk_proj = torch.nn.Linear(spk_channels, spk_enc_channels)
        self.token_emb = token_emb
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(encoder.output_size, spk_enc_channels)
        self.estimator = estimator
        # Random x0, maximum of 600s
        self.register_buffer(
            "x0",
            torch.randn([1, self.estimator.out_channels, 50 * 600]),
            persistent=False,
        )

    def _euler(
        self,
        x0: torch.Tensor,
        c: torch.Tensor,
        n_timesteps: int = 10,
    ):
        # time steps
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(x0)
        # cosine time schduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        # euler solver
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        xt = x0
        for step in range(1, len(t_span)):
            # pack input
            x_in = torch.cat([xt, xt], dim=0)
            c_in = torch.cat([c, torch.zeros_like(c)], dim=0)
            t_in = torch.cat([t, t], dim=0)

            # model call
            with torch.no_grad():
                vt = self.estimator.forward(x_in, c_in, t_in)
            # cfg
            vt_cond, vt_cfg = vt.chunk(2, dim=0)
            vt = (1.0 + self.infer_cfg_rate) * vt_cond - self.infer_cfg_rate * vt_cfg

            xt = xt + dt * vt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return xt

    def inference(
        self,
        prompt_token: torch.Tensor,
        prompt_xvec: torch.Tensor,
        prompt_feat: torch.Tensor,
        token: torch.Tensor,
    ):
        # NOTE align prompt_token, prompt_feat in advance

        # Spk condition
        embedding = F.normalize(prompt_xvec, dim=1)
        spks = self.spk_proj(embedding)

        # Token condition
        token = torch.concat([prompt_token, token], dim=1)
        xs = self.token_emb(token)

        xs_lens = torch.tensor([xs.shape[1]]).to(token)
        xs = self.encoder(xs, xs_lens)
        mu = self.encoder_proj(xs)

        # Mel context
        ctx = torch.zeros_like(mu)
        ctx[:, : prompt_feat.shape[1]] = prompt_feat

        # Compose condition
        cond = mu.transpose(1, 2)
        ctx = ctx.transpose(1, 2)
        spks = repeat(spks, "b c -> b c t", t=cond.shape[-1])
        cond = pack([cond, spks, ctx], "b * t")[0]

        # FM inference
        x0 = self.x0[..., : mu.shape[1]]
        x1 = self._euler(x0, cond, n_timesteps=10)

        feat = x1.transpose(1, 2)[:, prompt_feat.shape[1] :]
        return feat
