import torch
import torch.nn as nn
import torch.nn.functional as F

from my_gpt.layers.transformer_block import TransformerBlock
from my_gpt.layers.layer_norm import LayerNorm

class GPT(nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            num_heads,
            num_layers,
            block_size,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.block_size = block_size

        # -------------------------------------
        # Token Embeddings
        # -------------------------------------

        self.token_embedding = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.02
        )

        # -------------------------------------
        # Positional Embeddings
        # -------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(block_size, embedding_dim) * 0.02
        )

        # -------------------------------------
        # Transformer Blocks
        # -------------------------------------

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                block_size=block_size
            ) for _ in range(num_layers)
        ])

        # -------------------------------------
        # Final Layer Norm
        # -------------------------------------

        self.final_norm = LayerNorm(embedding_dim)

        # -------------------------------------
        # Vocab Projection
        # -------------------------------------

        self.W_vocab = nn.Parameter(
            torch.randn(embedding_dim, vocab_size) * 0.02
        )

        self.b_vocab = nn.Parameter(
            torch.zeros(vocab_size)
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {T}, block size is only {self.block_size}")

        # -------------------------------------
        # Token Embeddings
        # -------------------------------------

        token_emb = self.token_embedding[idx]  # (B, T, embedding_dim)

        # -------------------------------------
        # Positional Embeddings
        # -------------------------------------

        positions = torch.arange(
            T, 
            device=idx.device
        )

        pos_emb = self.position_embedding[positions]  # (T, embedding_dim)

        # -------------------------------------
        # Combine Token and Positional Embeddings
        # -------------------------------------

        x = token_emb + pos_emb  # (B, T, embedding_dim)

        # -------------------------------------
        # Transformer Blocks
        # -------------------------------------

        for block in self.blocks:
            x = block(x)  # (B, T, embedding_dim)

        # -------------------------------------
        # Final Layer Norm
        # -------------------------------------

        x = self.final_norm(x)  # (B, T, embedding_dim)

        # -------------------------------------
        # Vocab Projection
        # -------------------------------------

        logits = x @ self.W_vocab + self.b_vocab  # (B, T, vocab_size)

        # -------------------------------------
        # Loss Calculation
        # -------------------------------------

        loss = None

        if targets is not None:

            B, T, V = logits.shape

            logits_flat = logits.reshape(B * T, V)

            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        for _ in range(max_new_tokens):

            # On garde uniquement les derniers block_size tokens
            idx_cond = idx[:, -self.block_size:]

            # On calcule les logits pour les tokens suivants (Forward pass)
            logits, _ = self(idx_cond)
        
            # On ne garde que les logits du dernier token
            logits = logits[:, -1, :]  # (B, vocab_size)

            # On applique softmax pour obtenir les probabilités, en tenant compte de la température
            probs = F.softmax( logits / temperature, dim=-1)  # (B, vocab_size)

            # On échantillonne le prochain token à partir de la distribution de probabilité
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # On ajoute le token échantillonné à la séquence d'entrée
            idx = torch.cat((idx, next_token), dim=1)  # (B, T+1)

        return idx

if __name__ == "__main__":

    vocab_size = 40
    embedding_dim = 64
    num_heads = 4
    num_layers = 3
    block_size = 16

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size
    )

    # Test forward pass
    B = 4
    T = 16

    idx = torch.randint(0, vocab_size, (B, T))

    targets = torch.randint(0, vocab_size, (B, T))

    logits, loss = model(idx, targets)

    print("Logits shape:", logits.shape)  # Expected: (B, T, vocab_size)
    print("Loss:", loss.item())
