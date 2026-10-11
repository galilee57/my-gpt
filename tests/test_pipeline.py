"""Vérifier le modèle et les commandes après réorganisation."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

import generate
import train
from my_gpt.models.bigram import BigramLanguageModel
from my_gpt.models.gpt import GPT
from my_gpt.tokenizer import CharTokenizer


class PipelineTests(unittest.TestCase):
    def test_models_gradients_and_causality(self):
        torch.manual_seed(42)
        model = GPT(12, 8, 2, 1, 8)
        idx = torch.randint(0, 12, (2, 8))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (2, 8, 12))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        changed = idx.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 12
        self.assertTrue(torch.allclose(model(idx)[0][:, :4], model(changed)[0][:, :4]))
        output = model.generate(idx, 3, temperature=0.7)
        self.assertEqual(output.shape, (2, 11))
        self.assertTrue(torch.equal(output[:, :8], idx))
        _, bigram_loss = BigramLanguageModel(12)(idx, idx)
        self.assertTrue(torch.isfinite(bigram_loss))

    def test_named_run_checkpoint_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / 'corpus.txt'
            corpus.write_text('bonjour le monde. un exemple de texte.\n' * 30)
            options = ['--name', 'smoke', '--corpus', str(corpus), '--output-dir', directory,
                       '--embedding-dim', '8', '--num-layers', '1', '--num-heads', '2',
                       '--block-size', '8', '--batch-size', '2', '--max-steps', '3',
                       '--eval-interval', '2', '--eval-iters', '1', '--generate-interval', '2',
                       '--generate-tokens', '2']
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(options)
            run = Path(directory) / 'runs/smoke'
            checkpoint = run / 'best_model.pt'
            config = json.loads((run / 'config.json').read_text())
            self.assertEqual(Path(config['checkpoint_path']), checkpoint.resolve())
            with (run / 'metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row['step']) for row in rows], [0, 2, 3])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            best = min(rows, key=lambda row: float(row['val_loss']))
            self.assertEqual(saved['step'], int(best['step']))
            tokenizer = CharTokenizer('', charset=''.join(saved['tokenizer_chars']))
            self.assertEqual(tokenizer.chars, saved['tokenizer_chars'])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                generate.main(['bonjour ', '--tokens', '3', '--checkpoint', str(checkpoint),
                               '--temperature', '0.7'])
            self.assertIn('bonjour ', output.getvalue())
            self.assertTrue((run / 'samples.txt').is_file())
            original = checkpoint.read_bytes()
            with self.assertRaises(FileExistsError):
                train.main(options)
            self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
