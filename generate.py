import argparse
from pathlib import Path
import torch

from my_gpt.tokenizer import CharTokenizer
from my_gpt.models.gpt import GPT



def main(argv=None):
    # --------------------------------
    # Hyperparameters
    # --------------------------------

    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")

    parser.add_argument(
        "prompt",
        type=str,
        help="The initial text prompt to start the generation."
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
        help="The number of tokens to generate."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint to load."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature for sampling."
    )

    args = parser.parse_args(argv)

    if not args.prompt:
        parser.error("Le prompt ne doit pas etre vide.")
    if args.tokens < 0:
        parser.error("--tokens doit etre positif ou nul.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint introuvable : {args.checkpoint}. Lancez train.py ou utilisez --checkpoint.")

    # --------------------------------
    # Device configuration
    # --------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --------------------------------
    # Load the model checkpoint
    # --------------------------------

    checkpoint = torch.load(
        args.checkpoint, 
        map_location="cpu",
        weights_only=True
    )

    # --------------------------------
    # Rebuild the tokenizer and model
    # --------------------------------

    tokenizer = CharTokenizer(
        text="",
        charset="".join(checkpoint["tokenizer_chars"])
    )

    if tokenizer.chars != checkpoint["tokenizer_chars"]:
        parser.error("L'ordre des caracteres du checkpoint est incompatible avec le tokenizer.")

    config = checkpoint["model_config"]
    if tokenizer.vocab_size != config["vocab_size"]:
        parser.error("Le vocabulaire du tokenizer ne correspond pas a celui du modele.")

    model = GPT(**config)

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    model.eval()

    # --------------------------------
    # Prepare the input prompt
    # --------------------------------

    try:
        tokens = tokenizer.encode(args.prompt)
    except ValueError as error:
        parser.error(str(error))

    idx = torch.tensor(
        [tokens], 
        dtype=torch.long,
        device=device
    )

    # --------------------------------
    # Generate text
    # --------------------------------

    with torch.inference_mode():

        generated_tokens = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature
        )

    # ---------------------------------
    # Decode and print the generated text
    # --------------------------------- 

    result = tokenizer.decode(generated_tokens[0].tolist())

    print("Generated Text:\n")
    print(result)


if __name__ == "__main__":
    main()
