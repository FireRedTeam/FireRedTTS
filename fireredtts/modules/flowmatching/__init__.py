from .estimator_dit import DiT
from .upsample_encoder import UpsampleConformerEncoder
from .flow import CausalFmWithSpkCtx, DualEmbedding


class FlowToken2Mel(CausalFmWithSpkCtx):
    def __init__(self, config):
        token_emb = DualEmbedding(**config['token_emb'])
        encoder = UpsampleConformerEncoder(**config['encoder'])
        estimator = DiT(**config['estimator'])
        super().__init__(
            spk_channels=config['spk_channels'],
            spk_enc_channels=config['spk_enc_channels'],
            infer_cfg_rate=config['infer_cfg_rate'],
            token_emb=token_emb,
            encoder=encoder,
            estimator=estimator,
        )