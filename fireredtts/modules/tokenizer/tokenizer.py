import os
import torch
from fireredtts.modules.tokenizer.whisper_tokenizer import get_tokenizer


DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../data/tokenizer.json"
)

class VoiceBpeTokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer(multilingual=True)

    def encode(self, lines):
        return [self.tokenizer.encode(t) for t in lines]

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        text = self.tokenizer.decode(seq)
        return text

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return self.tokenizer.get_vocab_size()


if __name__ == "__main__":
    tok = VoiceBpeTokenizer()
    codes = tok.encode("我、真是My love is Mr. TTS. hello USA啊？谢谢你world！")
    for line in codes:
        print([tok.decode([c]) for c in line])
