import torch
import torch.nn as nn

from my_gpt.layers.feed_forward import FeedForward
from my_gpt.layers.layer_norm import LayerNorm
from my_gpt.layers.multihead_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        block_size
    ):
        super().__init__()

        self.ln1 = LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            block_size=block_size
        )

        self.ln2 = LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.feed_forward(
            self.ln2(x)
        )

        return x


if __name__ == "__main__":

    B = 2
    T = 8
    C = 32

    x = torch.randn(
        B,
        T,
        C
    )

    block = TransformerBlock(
        embedding_dim=C,
        num_heads=4,
        block_size=T
    )

    out = block(x)

    print("Input :", x.shape)
    print("Wq:", block.attention.heads[0].W_q.shape)
    print("Wk:", block.attention.heads[0].W_k.shape)
    print("Wv:", block.attention.heads[0].W_v.shape)
    print("Output:", out.shape)
