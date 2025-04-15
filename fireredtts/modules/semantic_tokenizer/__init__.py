import os
import torch

from .hubert import HuBERT
from .semantic_tokenizer import SemanticVQVAE


class SemanticTokenizer:

    def __init__(self, config, path):
        self.model = SemanticVQVAE(**config)
        self.model.load_state_dict(
            torch.load(os.path.join(path, "codec.bin"), map_location="cpu"), strict=True
        )

        hubert = HuBERT(os.path.join(path, "hubert.pt"))
        for name, param in hubert.named_parameters():
            param.requires_grad = False
        self.model.ssl_extractor = hubert

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def __call__(self, wavs, wav_lengths):
        tokens, token_lengths, spk_embeddings = self.extract(wavs, wav_lengths)
        return tokens, token_lengths, spk_embeddings

    def extract(self, wavs, wav_lengths):
        saved_features = self.model.extract_speech_tokens(wavs, wav_lengths)

        tokens = saved_features["token"]
        token_lengths = saved_features["token_length"]
        spk_embeddings = saved_features["spk"]

        return tokens, token_lengths, spk_embeddings
