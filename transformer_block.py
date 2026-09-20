import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):

    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_size = head_size

        # Wq
        self.W_q = nn.Parameter(torch.randn(embedding_dim, head_size) * 0.02)

        # Wk
        self.W_k = nn.Parameter(torch.randn(embedding_dim, head_size) * 0.02)

        # Wv
        self.W_v = nn.Parameter(torch.randn(embedding_dim, head_size) * 0.02)

        # Causal mask to ensure that attention is only applied to previous tokens
        mask = torch.tril(torch.ones(block_size, block_size))

        self.register_buffer("causal_mask", mask)

    def forward(self, x):

        B, T, C = x.shape

        # Compute queries, keys, and values
        q = x @ self.W_q  # (B, T, head_size)
        k = x @ self.W_k     # (B, T, head_size)
        v = x @ self.W_v   # (B, T, head_size)

        # Compute attention scores
        scores = q @ k.transpose(-2, -1)

        # Scale scores by the square root of the head size
        scores = scores / (self.head_size ** 0.5)

        # Apply causal mask
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute the output of the attention head
        out = attention_weights @ v  # (B, T, head_size)

        return out


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


class LayerNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(embedding_dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        variance = (
            (x - mean) ** 2
        ).mean(
            dim=-1,
            keepdim=True
        )

        x_normalized = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        out = (
            self.gamma * x_normalized
            + self.beta
        )

        return out


class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        hidden_dim = 4 * embedding_dim

        self.W1 = nn.Parameter(
            torch.randn(
                embedding_dim,
                hidden_dim
            ) * 0.02
        )

        self.b1 = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.W2 = nn.Parameter(
            torch.randn(
                hidden_dim,
                embedding_dim
            ) * 0.02
        )

        self.b2 = nn.Parameter(
            torch.zeros(embedding_dim)
        )

    def forward(self, x):

        x = x @ self.W1 + self.b1

        x = torch.relu(x)

        x = x @ self.W2 + self.b2

        return x

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
    print("Output:", out.shape)