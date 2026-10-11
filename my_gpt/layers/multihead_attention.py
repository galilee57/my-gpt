import torch
import torch.nn as nn

from my_gpt.layers.self_attention_head import SelfAttentionHead


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_size = (
            embedding_dim // num_heads
        )

        self.heads = nn.ModuleList([
            SelfAttentionHead(
                embedding_dim=embedding_dim,
                head_size=self.head_size,
                block_size=block_size
            )
            for _ in range(num_heads)
        ])

        self.W_o = nn.Parameter(
            torch.randn(
                embedding_dim,
                embedding_dim
            ) * 0.02
        )

    def forward(self, x):

        outputs = []

        for head in self.heads:
            head_output = head(x)
            outputs.append(head_output)

        concatenated = torch.cat(
            outputs,
            dim=-1
        )

        out = concatenated @ self.W_o

        return out
