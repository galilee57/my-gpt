# Corpus généraliste synthétique — 11 octobre 2026

## Contenu

`processed/generaliste1m.txt` contient **1 000 264 caractères**, environ **159 674 mots** et **1 525 documents courts**. Le vocabulaire contient 50 caractères distincts. Tout est en minuscules, sans accents, en ASCII, comme les corpus précédents.

| Format | Nombre de documents |
| --- | ---: |
| Situations, récits, dialogues, comptes rendus et consignes | 743 |
| Explications, fiches et questions-réponses | 484 |
| Exercices numériques avec solution calculée | 298 |

Les 50 thèmes explicatifs couvrent notamment la nature, la physique élémentaire, la géographie, l'histoire et les sources, la langue, la culture, les mathématiques, l'informatique, l'organisation et le raisonnement critique. Les 50 familles de situations concernent le travail, les associations, la maison, les déplacements, les loisirs et l'apprentissage.

Les connaissances et situations de base ont été rédigées pour ce projet. Les noms, les cadres narratifs et les formulations sont combinés de façon déterministe ; les réponses arithmétiques sont calculées en Python. Le corpus ne reproduit aucun livre téléchargé et ne contient pas d'actualité ou de conseils professionnels. Les documents sont mélangés pour ne pas réserver un seul thème aux derniers 10 %.

## Usage

Depuis la racine du projet, avec un nouveau nom d'expérience :

```bash
python train.py --name generaliste1m_dim64_layers2_ctx128 \
  --corpus data/processed/generaliste1m.txt \
  --embedding-dim 64 --num-layers 2 --num-heads 4 \
  --block-size 128 --max-steps 5000 --seed 42
```

Puis :

```bash
python generate.py "comment organiser une reunion ? " \
  --tokens 200 \
  --checkpoint runs/generaliste1m_dim64_layers2_ctx128/best_model.pt
```

Autres débuts de texte à comparer :

```text
le matin, lea
question : pourquoi
pour preparer une recette,
un programme
```

Ce corpus entraîne la continuation de texte. La présence de questions-réponses ne suffit pas à transformer le modèle en assistant capable de suivre toutes les consignes. L'entraînement complet n'a pas été lancé lors de la préparation.

## Diversité réelle et limites

La taille d'un million de caractères est obtenue par **variation de structures rédigées**, et non par un million de caractères de rédaction indépendante. Il n'y a aucun document intégralement dupliqué, mais les formulations, les explications et les faits sont réutilisés. Le contrôle compte 12 689 segments de phrase, dont 4 679 distincts ; 8 010 occurrences répètent un segment précédent. Ce comptage est indicatif : il découpe sur la ponctuation et ne mesure pas à lui seul la qualité.

Ce corpus est utile pour tester plusieurs registres et des relations simples entre phrases. Il ne fournit ni la richesse linguistique d'un grand corpus humain, ni une couverture exhaustive des connaissances. Une perte faible peut surtout refléter l'apprentissage des structures récurrentes. Il serait trompeur d'en déduire une compétence généraliste.

`train.py` réserve automatiquement les derniers 10 % à la validation. Des variantes proches peuvent donc apparaître dans les deux partitions, ce qui facilite la validation. Pour mesurer une généralisation réelle, il faudra compléter l'évaluation avec des textes humains ou des familles de situations entièrement tenus à l'écart de l'entraînement. Ne pas comparer directement cette perte avec celle de Ségur, puisque le texte évalué et son vocabulaire diffèrent.

Les exercices numériques sont exacts dans les hypothèses énoncées. Leur présence ne garantit pas que le modèle apprendra à calculer avec de nouveaux nombres.

## Reproduction et traçabilité

Le script `scripts/prepare_generaliste.py` contient les textes de base et la procédure de génération. Il n'utilise aucune API ni connexion réseau. La graine est `20261011` ; le fichier `processed/generaliste1m_metadata.json` enregistre les tailles, les types de documents, les limites et les empreintes SHA-256 du corpus et du script.

Le script refuse d'écraser un corpus existant. Pour reproduire ailleurs sans toucher aux données du projet, copier le script dans un répertoire ayant la même structure `scripts/` et `data/processed/`, puis exécuter :

```bash
python scripts/prepare_generaliste.py
```

Les corpus Ségur et le corpus synthétique initial sont conservés.
