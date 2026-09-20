import torch

from tokenizer import CharTokenizer
from dataset import get_batch
from model import BigramLanguageModel

# --------------------
# Hyperparameters
# --------------------

batch_size = 32
block_size = 8

learning_rate = 1e-3
steps = 5000

# --------------------
# Load and preprocess data
# --------------------

with open("data/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

data = torch.tensor(
    tokenizer.encode(text=text),
    dtype=torch.long
)

# --------------------
# Initialize model and optimizer
# --------------------

model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --------------------
# Training loop
# --------------------

for step in range(steps):
    # Get a batch of data
    x, y = get_batch(data, block_size, batch_size)

    # Forward pass
    logits, loss = model(x, targets=y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}")