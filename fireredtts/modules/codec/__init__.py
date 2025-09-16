import torch
import torchaudio
from functools import cached_property
from typing import Dict, Tuple
from fireredtts.modules.codec.semantic_tokenizer import SemanticVQVAE
from fireredtts.modules.codec.hubert import HuBERT


class SemanticTokenizer(torch.nn.Module):
    def __init__(self, hubert:HuBERT, model:SemanticVQVAE):
        super().__init__()
        self.model = model
        self.model.ssl_extractor = hubert
    
    @classmethod
    def from_pretrained(cls, config:Dict, hubert_path:str, codec_path:str)->'SemanticTokenizer':
        hubert = HuBERT(hubert_path)
        for p in hubert.parameters(): p.requires_grad = False
        model = SemanticVQVAE(**config)
        model.load_state_dict(torch.load(codec_path, map_location='cpu'))
        model.eval()
        return cls(hubert, model)

    @cached_property
    def device(self):
        return next(self.model.parameters()).device

    def __call__(self, audio:torch.Tensor, audio_sr:int)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: shape (1, t), audio tensor, on cuda or cpu
            audio_sr: audio sampling rate
        Returns:
            tokens: shape (1, l)
            spk: shape (1, c=512)
        """
        if audio_sr != 16000:
            audio = torchaudio.functional.resample(audio, audio_sr, 16000)
            audio_sr = 16000
        audio_length = torch.tensor([audio.shape[1]], dtype=torch.long)

        audio, audio_length = audio.to(self.device), audio_length.to(self.device)
        codec_features = self.model.extract_speech_tokens(audio, audio_length)
        tokens, spk = codec_features["token"], codec_features['spk']
        return tokens, spk

