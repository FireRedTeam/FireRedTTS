from fireredtts.modules.flow_v2.embedding import DualEmbedding
from fireredtts.modules.flow_v2.upsample_encoder import UpsampleConformerEncoder
from fireredtts.modules.flow_v2.estimator_dit import DiT
from fireredtts.modules.flow_v2.flow import CausalFmWithSpkCtx
from fireredtts.modules.flow_v2.mel_spectrogram import MelExtractor


def get_flow_frontend(flow_config):
    flow = CausalFmWithSpkCtx(
        spk_channels=flow_config["spk_channels"],
        spk_enc_channels=flow_config["spk_enc_channels"],
        infer_cfg_rate=flow_config["infer_cfg_rate"],
        token_emb=DualEmbedding(**flow_config["token_emb"]),
        encoder=UpsampleConformerEncoder(**flow_config["encoder"]),
        estimator=DiT(**flow_config["DiT"]),
    )
    return flow
