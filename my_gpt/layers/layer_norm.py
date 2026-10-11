import torch
import torch.nn as nn


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
