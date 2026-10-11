from pathlib import Path
from typing import List
import torch

class CharTokenizer:
    def __init__(self, text: str, charset: str = None):

        # If charset is provided, use it; otherwise, derive charset from the text
        if charset is None:
            self.chars = sorted(list(set(text)))
        elif text is not None:
            self.chars = sorted(list(set(charset)))
        else:
            raise ValueError("Either text or charset must be provided.")

        self.vocab_size = len(self.chars)

        # string -> int
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}

        # int -> string
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:

        unknown = set(text) - set(self.stoi)

        if unknown:
            raise ValueError(f"Unknown characters found in text: {unknown}")

        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: List[int]) -> str:

        return "".join([self.itos[token] for token in tokens])

if __name__ == "__main__":

    corpus_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "corpus10k.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
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