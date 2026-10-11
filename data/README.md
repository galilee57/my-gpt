# Données du projet

- `raw/` : réservé aux futurs textes sources avant nettoyage. Aucun texte source supplémentaire n'a été téléchargé lors de la réorganisation.
- `processed/` : corpus prêts à l'entraînement et fichiers de métadonnées associés.
- [Corpus Ségur](README_SEGUR.md) : source littéraire, préparation et limites de comparaison.
- [Corpus synthétique généraliste](README_GENERALISTE.md) : génération, reproductibilité et limites de diversité.

| Corpus | Rôle |
| --- | --- |
| `processed/corpus10k.txt` | Premier corpus synthétique, 10 622 caractères ; corpus par défaut de `train.py`. |
| `processed/segur10k.txt` | Extrait de Ségur, 10 009 caractères. |
| `processed/segur100k.txt` | Extrait de Ségur, 100 031 caractères. |
| `processed/generaliste1m.txt` | Corpus synthétique à structures contrôlées, 1 000 264 caractères. |

Les métadonnées d'origine sont conservées. Le champ `generator_sha256` du corpus généraliste correspond au script au moment de la création ; son chemin de sortie a depuis été adapté à `data/processed/`. Le contenu du corpus reste identique.

Depuis la racine du projet :

```bash
python train.py --name segur100k_nouvel_essai --corpus data/processed/segur100k.txt --embedding-dim 64 --num-layers 2 --block-size 128
```
