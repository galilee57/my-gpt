# Corpus Ségur — préparation du 11 octobre 2026

Source : **Les Petites Filles modèles**, comtesse de Ségur, [Project Gutenberg, livre 15059](https://www.gutenberg.org/ebooks/15059). Le [texte brut](https://www.gutenberg.org/cache/epub/15059/pg15059.txt) a servi à préparer ces extraits.

| Fichier | Caractères | Mots, séparés par des espaces | Caractères distincts |
| --- | ---: | ---: | ---: |
| `processed/segur10k.txt` | 10 009 | 1 638 | 37 |
| `processed/segur100k.txt` | 100 031 | 16 277 | 42 |

Les tailles comprennent les espaces et les retours à la ligne. Les extraits commencent au premier paragraphe du chapitre I. Le corpus 10k est un préfixe du corpus 100k ; chacun se termine à la première fin de paragraphe après la taille demandée, sans couper un mot ou une phrase.

## Nettoyage commun

- Retrait des en-têtes et pieds de page Gutenberg, des crédits, de la table des matières, de la préface et des titres de chapitres.
- Minuscules, suppression des accents et conversion des ligatures en `oe` et `ae`.
- Normalisation des guillemets, apostrophes et tirets en ASCII ; suppression des marqueurs d'italique `_`.
- Réunion des lignes coupées par la mise en page, en conservant les paragraphes, les noms des locuteurs et la ponctuation.
- Aucun ajout de phrases synthétiques, aucune répétition pour atteindre la taille cible.

Les empreintes du téléchargement et des deux corpus, ainsi que leurs statistiques, sont conservées dans `processed/segur_metadata.json`. Le fichier synthétique existant `processed/corpus10k.txt` est conservé séparément.

## Entraîner

Depuis la racine du projet :

```bash
python train.py --name segur10k --corpus data/processed/segur10k.txt --seed 42
python train.py --name segur100k --corpus data/processed/segur100k.txt --seed 42
```

Puis, par exemple :

```bash
python generate.py "il etait une fois " --tokens 200 --checkpoint runs/segur10k/best_model.pt
python generate.py "il etait une fois " --tokens 200 --checkpoint runs/segur100k/best_model.pt
```

## Limites de la comparaison actuelle

`train.py` réserve les derniers 10 % de chaque fichier à la validation. Les passages de validation diffèrent donc entre les deux expériences. De plus, la validation du petit corpus est incluse dans l'entraînement du grand corpus : ce ne sont pas des scores sur un même jeu de test indépendant. Pour une comparaison rigoureuse, prévoir ultérieurement un troisième passage tenu à l'écart des deux entraînements et l'utiliser pour évaluer les deux modèles.

Le tokenizer reconstruit son vocabulaire à partir de chaque corpus : 37 caractères pour le petit, 42 pour le grand. La taille des embeddings et de la projection varie donc légèrement, même si les autres hyperparamètres restent identiques. Consigner cette différence dans les résultats. Une comparaison avec exactement la même architecture demanderait un vocabulaire partagé.
