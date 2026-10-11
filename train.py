"""Entraîner un GPT et conserver les résultats d'une expérience nommée."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

from my_gpt.dataset import get_batch
from my_gpt.models.gpt import GPT
from my_gpt.tokenizer import CharTokenizer

project_dir = Path(__file__).resolve().parent


def model_metrics(model):
    """Count stored parameters once, including their actual dtype size."""
    parameters = list(model.parameters())
    return {
        "total_parameters": sum(p.numel() for p in parameters),
        "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in parameters if not p.requires_grad),
        "parameters_by_component": {
            name: sum(p.numel() for key, p in model.named_parameters()
                      if key.split(".")[0] == name)
            for name in dict.fromkeys(key.split(".")[0] for key, _ in model.named_parameters())
        },
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
    }


def write_config(path, experiment):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=datetime.now(timezone.utc).strftime("gpt_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--corpus", type=Path, default=project_dir / "data" / "processed" / "corpus10k.txt")
    parser.add_argument("--output-dir", type=Path, default=project_dir)
    for name, default in (
        ("batch-size", 32), ("block-size", 64), ("embedding-dim", 128),
        ("num-heads", 4), ("num-layers", 4), ("max-steps", 5000),
        ("eval-interval", 250), ("eval-iters", 100),
        ("generate-interval", 500), ("generate-tokens", 200),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("--name doit contenir uniquement des lettres, chiffres, tirets ou underscores.")
    for name in ("batch_size", "block_size", "embedding_dim", "num_heads", "num_layers",
                 "max_steps", "eval_interval", "eval_iters"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} doit etre strictement positif.")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate doit etre un nombre positif fini.")
    if args.generate_interval < 0 or args.generate_tokens < 0:
        parser.error("Les options de generation doivent etre positives ou nulles.")
    if args.embedding_dim % args.num_heads:
        parser.error("--embedding-dim doit etre divisible par --num-heads.")
    if not args.corpus.is_file():
        parser.error(f"Corpus introuvable : {args.corpus}")
    return args


def save_checkpoint(path, step, losses, model, model_config, tokenizer, optimizer, experiment):
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_chars": tokenizer.chars,
        "step": step,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "experiment": experiment,
    }, path)
    print(f"Checkpoint saved: {path}")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Read the exact bytes whose fingerprint will be recorded.
    corpus_bytes = args.corpus.read_bytes()
    text = corpus_bytes.decode("utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data, val_data = data[:split_idx], data[split_idx:]
    if min(len(train_data), len(val_data)) <= args.block_size + 1:
        raise ValueError(
            f"Chaque partition doit contenir au moins {args.block_size + 2} tokens. "
            "Reduisez --block-size ou ajoutez du texte."
        )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "block_size": args.block_size,
    }
    model = GPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    run_dir = args.output_dir / "runs" / args.name
    checkpoint_path = run_dir / "best_model.pt"
    if checkpoint_path.exists() or run_dir.exists():
        raise FileExistsError(
            f"L'experience {args.name!r} existe deja. Choisissez un nouveau --name pour conserver les resultats."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    experiment = {
        **{key: str(value.resolve()) if isinstance(value, Path) else value
           for key, value in vars(args).items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_characters": len(text),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "model_config": model_config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_metrics": model_metrics(model),
        "training_budget": {
            "tokens_per_update": args.batch_size * args.block_size,
            "planned_training_tokens": args.max_steps * args.batch_size * args.block_size,
            "planned_equivalent_corpus_passes": (
                args.max_steps * args.batch_size * args.block_size / len(train_data)
            ),
            "evaluation_tokens_per_split": args.eval_iters * args.batch_size * args.block_size,
            "parameters_per_train_token": sum(p.numel() for p in model.parameters()) / len(train_data),
            "uniform_baseline_loss_nats": math.log(tokenizer.vocab_size),
        },
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    write_config(run_dir / "config.json", experiment)
    print(f"Experiment: {args.name}; device: {device}; corpus: {len(text)} characters")
    print(f"Parameters: {experiment['parameter_count']:,}")
    print(f"Metrics: {run_dir / 'metrics.csv'}")

    def batch(split):
        return get_batch(train_data, val_data, split, args.block_size, args.batch_size, device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        try:
            for split in ("train", "val"):
                total = 0.0
                for _ in range(args.eval_iters):
                    x, y = batch(split)
                    _, loss = model(x, targets=y)
                    total += loss.item()
                losses[split] = total / args.eval_iters
        finally:
            model.train()
        return losses

    best_val_loss = float("inf")
    best_step = None
    started = time.perf_counter()
    # step counts completed optimizer updates; evaluate the final model too.
    with (run_dir / "metrics.csv").open("x", newline="", encoding="utf-8") as metrics:
        writer = csv.DictWriter(metrics, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        metrics.flush()
        for step in range(args.max_steps + 1):
            if step % args.eval_interval == 0 or step == args.max_steps:
                losses = estimate_loss()
                if not all(math.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Perte non finie : verifiez les reglages d'entrainement.")
                improved = losses["val"] < best_val_loss
                if improved:
                    best_val_loss, best_step = losses["val"], step
                elapsed = time.perf_counter() - started
                seen_tokens = step * args.batch_size * args.block_size
                experiment["training_metrics"] = {
                    "completed_updates": step,
                    "training_tokens_seen": seen_tokens,
                    "equivalent_corpus_passes": seen_tokens / len(train_data),
                    "elapsed_seconds_at_evaluation": elapsed,
                    "end_to_end_training_tokens_per_second": seen_tokens / elapsed,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "val_minus_train_loss": losses["val"] - losses["train"],
                    "val_bits_per_character": losses["val"] / math.log(2),
                    "val_perplexity": math.exp(losses["val"]) if losses["val"] < 709 else None,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                }
                write_config(run_dir / "config.json", experiment)
                writer.writerow({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})
                metrics.flush()
                print(f"Step {step}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
                if improved:
                    save_checkpoint(checkpoint_path, step, losses, model, model_config,
                                    tokenizer, optimizer, experiment)

            if args.generate_interval and step > 0 and step % args.generate_interval == 0:
                model.eval()
                with torch.inference_mode():
                    start = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
                    generated = model.generate(start, max_new_tokens=args.generate_tokens)
                    sample = tokenizer.decode(generated[0].tolist())
                model.train()
                with (run_dir / "samples.txt").open("a", encoding="utf-8") as samples:
                    samples.write(f"Step {step}\n{sample}\n\n")
                print(f"Sample text at step {step}:\n{sample}\n")

            if step == args.max_steps:
                break
            x, y = batch("train")
            _, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
