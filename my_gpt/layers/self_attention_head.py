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
