import json
import torch
import torch.nn as nn
from typing import Dict
from fireredtts.modules.bigvgan import get_bigvgan_backend

from fireredtts.modules.flow_v2 import get_flow_frontend as get_flow_v2_frontend
from fireredtts.modules.flow_v2 import MelExtractor as MelSpectrogramExtractorV2


"""Flowmatching-v2.
"""


class Token2Wav_v2(nn.Module):
    def __init__(self, flow: nn.Module, generator: nn.Module):
        super().__init__()
        self.flow = flow
        self.generator = generator

    @classmethod
    def from_pretrained(cls, config: Dict, ckpt_path: str) -> "Token2Wav_v2":
        flow = get_flow_v2_frontend(config["flow"])
        bigvgan = get_bigvgan_backend(config["bigvgan"])
        token2wav = cls(flow, bigvgan)
        token2wav.eval()
        token2wav.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        token2wav.generator.remove_weight_norm()
        return token2wav

    def inference(
        self,
        prompt_token: torch.Tensor,
        prompt_xvec: torch.Tensor,
        prompt_mel: torch.Tensor,
        token: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            mel = self.flow.inference(
                prompt_token=prompt_token,
                prompt_xvec=prompt_xvec,
                prompt_feat=prompt_mel,
                token=token,
            )
            audio = self.generator(mel.transpose(1, 2))
        return audio.view(1, -1)
