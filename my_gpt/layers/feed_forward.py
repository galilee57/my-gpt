import torch
import torch.nn as nn


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
