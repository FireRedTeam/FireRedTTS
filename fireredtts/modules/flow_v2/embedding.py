import torch


class DualEmbedding(torch.nn.Module):
    def __init__(
        self, 
        channels:int=512,
    ):
        super().__init__()
        self.codebook = torch.nn.ModuleList([
            torch.nn.Embedding(128, 128),
            torch.nn.Embedding(128, 128),
        ])
        self.out_proj = torch.nn.Linear(256, channels)

    def forward(self, tokens):
        """
        Args:
            tokens: shape (b, t)
        Returns:
            token_embs: shape (b, t, c)
        """
        token_embs = torch.cat(
            [self.codebook[0](tokens % 128), self.codebook[1](tokens // 128)], dim=-1
        )
        token_embs = self.out_proj(token_embs)
        return token_embs
