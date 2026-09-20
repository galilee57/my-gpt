from typing import List
import torch

class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))

        self.vocab_size = len(self.chars)

        # string -> int
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

        # int -> string
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos[token] for token in tokens])

if __name__ == "__main__":

    with open("data/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)

    print("Vocabulaire :", tokenizer.chars)
    print("Taille du vocabulaire :", tokenizer.vocab_size)

    # Test encoding and decoding
    sample_text = "bonjour"
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    tokens = tokenizer.encode(text=text)

    data = torch.tensor(
        tokens,
        dtype=torch.long
    )

    print(f"Data: {data}, Shape: {data.shape}, Dtype: {data.dtype}")