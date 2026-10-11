# my-gpt
Un petit GPT construit avec PyTorch, caractère par caractère, pour comprendre les modèles de langage.

## État actuel et progression

Le projet dispose désormais d’un GPT complet, d’un entraînement avec validation et checkpoints, et d’un script de génération. Le premier texte généré a été archivé le **10 octobre 2026** : il contient des mots reconnaissables, mais reste largement incohérent.

Le [journal de progression](docs/PROGRESSION.md) conserve le parcours, l’échantillon original, les mesures du checkpoint, l’analyse et une fiche pour comparer les prochaines expériences.

Il inclut aussi le retour de ChatGPT : des régularités locales semblent émerger, tandis que les mots et la syntaxe restent instables. La proposition de tester un corpus de 100 000 caractères y est consignée comme une prochaine expérience, sans garantie de qualité.

```bash
python train.py
python generate.py "il etait une fois" --tokens 1000 --checkpoint runs/segur100k/best_model.pt
```


## Organisation du projet

Chaque classe possède son propre fichier. Les composants sont séparés pour suivre les étapes de construction du modèle.

```text
my-gpt/
├── README.md
├── requirements.txt
├── train.py                      # Commande d'entraînement
├── generate.py                   # Commande de génération
├── my_gpt/                       # Code réutilisable
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bigram.py
│   │   └── gpt.py
│   └── layers/
│       ├── __init__.py
│       ├── self_attention_head.py
│       ├── multihead_attention.py
│       ├── layer_norm.py
│       ├── feed_forward.py
│       └── transformer_block.py
├── scripts/
│   └── prepare_generaliste.py
├── data/
│   ├── README.md
│   ├── raw/                      # Futurs textes sources
│   └── processed/                # Corpus et métadonnées
├── runs/
│   └── <nom_experience>/
│       ├── best_model.pt
│       ├── config.json
│       ├── metrics.csv
│       └── samples.txt
├── tests/
│   └── test_pipeline.py
└── docs/
    ├── PROGRESSION.md
    └── results/
```

Les imports suivent les dépendances : `MultiHeadAttention` utilise `SelfAttentionHead`, puis `TransformerBlock` assemble `MultiHeadAttention`, `LayerNorm` et `FeedForward`.

Depuis la racine du projet :

```bash
python -m my_gpt.tokenizer          # Démonstration du tokenizer
python -m my_gpt.layers.transformer_block  # Démonstration du bloc Transformer
python train.py              # Entraînement du GPT
```

## Au départ, on constitue un corpus simple.
Le LLM va être entrainé pour prédire la lettre suivante.
La prédiction du token suivant est aussi le principe utilisé par les modèles de langage autorégressifs. Ici, chaque token est un caractère ; les grands modèles utilisent souvent des fragments de mots.

Les exemples chiffrés des premières étapes ci-dessous décrivent le corpus initial de 153 caractères et son vocabulaire de 22 caractères. Le corpus actuel contient 10 622 caractères et son vocabulaire compte 33 caractères.

## 1ère étape : Tokenizer (my_gpt/tokenizer.py)
**Car un LLM ne connait par les lettres mais les nombres !**  
On construit un Tokenizer simple : chaque lettre / symbole du corpus a un token qui correspond à son index dans la liste constituée des signes présents.

+ Vocabulaire : ['\n', ' ', '.', 'a', 'b', 'c', 'd', 'e', 'g', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x']
+ Taille du vocabulaire : 22
+ Encoded: [4, 14, 13, 10, 14, 19, 16]
+ Decoded: bonjour

Avec torch, on transforme la liste en vecteur :  
+ Data: tensor([ 4, 14, 13, 10, 14, 19, 16, ... 11,  7,  1, 18,  7, 21, 18,  7,  2]), 
+ Shape: torch.Size([153])
+ vDtype: torch.int64

## 2ème étape : constitution du dataset pour l'entrainement (my_gpt/dataset.py)
Le GPT ne reçoit pas tout le texte : il faut lui définir une fenêtre `block_size=8` que le modèle regarde au maximum.  
On construit `x=chunk[:-1] et y=chunk[1:]` ce qui donne `x : b o n j o u r - y : o n j o u r`.
Par ce décalage, on crée en une séance plusieurs exercices : le modèle apprends simultanéménet b-> o, bo -> n, ...
C'est l'idée de **l'entrainement autoregressif** : avec un block_size=8 et batch_size=4, on aura 4 séances de 8 tokens.

## 3ème étape : construction du premier modèle BIGRAM (my_gpt/models/bigram.py)
Le Bigram modélise `P(x_{t+1} | x_t)` : il prédit le prochain caractère à partir du seul caractère courant.  
On crée une couche neuronale (nn.embedding) : si par ex vocab_size=30, PyTorch crée une matrice 30x30 : chaque token aura un vecteur de 30 nombres.  
Le modèle renvoie des LOGITS (score), qui sont ensuite transformés via SOFTMAX.  
La CROSSENTROPY permet de mesurer la distance entre prédiction et bonne réponse, puis on calcule le LOSS que l'on cherche à minimiser.

## 4ème étape : encodage d'une TETE de SELF-ATTENTION
On évite ici nn.MultiHeadAttention de Torch et les blocs Transformer tout faits.
L'objetif est de comprendre le fonctionnement en codant au maximum en Python pur.

+ Idée centrale : recréer le calcul de l'article $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$
+ La normalisation permet d'éviter que les produits scalaires tendent vers des valeurs divergentes. Ce qui détériorerait les gradients.
+ On applique un MASQUE CAUSAL pour empêcher chaque position de regarder le futur, c'est-à-dire que chaque token ne puisse pas regarder le token qui le suit. Pour ce faire, on remplace les zones interdites par -infinity (que le softmax transformera en probabilité 0)
+ On applique enfin un SOFTMAX.
+ On part d'un tenseur B (BATCH), T (LONGUEUR DE SEQUENCE), C (DIMENSION D'EMBEDDING)
+ On choisit une dimension de tête plus petite, pour maitriser le coût de calcul.
+ La création des matrices $ \ W_Q, W_K W_V \ $ se fera en Python pur (sans utiliser nn.Linear(..)).

Test :  
+ x :  torch.Size([2, 5, 8])
+ Wq :  torch.Size([8, 4])
+ output :  torch.Size([2, 5, 4])

## 5ème étape : construire la MULTI-HEAD
Au lieu d'avoir une seule tête qui produit une seule vision, on va en créer plusieurs en parallèle.
Chaque tête apprend avec ses propres matrices.
Pour obtenir le résultat, il faut concaténer leurs sorties, puis appliquer une projection finale.
Chaque tête produit `embedding_dim / num_heads` valeurs. Leur concaténation, puis la projection finale, donnent `embedding_dim` valeurs pour permettre la connexion résiduelle.  
Dans un LLM - avec GPU - on effectuerait les calculs en parallèle.

## 6ème étape : construire le TRANSFORMER BLOCK
L'architecture d'un bloc GPT moderne est : LayerNorm -> Multihead Self-Attention -> (connexion résiduelle) -> LayerNorm -> Feed Fordward -> (connexion résiduelle).  
La connexion résiduelle x + Attention(LN(x)) pemet de ne pas perdre l'information.  
L'attention vient s'AJOUTER aux tokens : elle ne les remplace pas.  ≠


## 7ème étape : assembler le GPT (my_gpt/models/gpt.py)

Chaque token reçoit un embedding de caractère et un embedding de position. Leurs sommes traversent les blocs Transformer, puis une normalisation finale et une projection vers le vocabulaire. L'entropie croisée compare les logits aux caractères suivants attendus.

## 8ème étape : entraîner, sauvegarder et générer

`train.py` sépare le corpus en 90 % pour l'entraînement et 10 % pour la validation. Le meilleur checkpoint évalué est enregistré dans `runs/gpt/best_model.pt`, avec la configuration du modèle, ses poids, les caractères du tokenizer, l'état de l'optimiseur, l'étape et la perte de validation.

`generate.py` recharge ce checkpoint et prolonge le prompt par tirages successifs. Avec la configuration actuelle, le modèle voit au maximum les 64 derniers caractères à chaque prédiction. `--tokens` indique le nombre de nouveaux caractères à produire. Utiliser des caractères présents dans le vocabulaire, actuellement des minuscules sans accents et la ponctuation du corpus.

Le [premier bilan](docs/PROGRESSION.md#expérience-001--première-génération-10-octobre-2026) distingue le fonctionnement de cette chaîne de la qualité linguistique encore limitée du résultat.

## Comparer plusieurs expériences

Nommer chaque essai et choisir son corpus et ses hyperparamètres en ligne de commande :

```bash
# Corpus actuel, environ 10k caractères
python train.py --name gpt10k --corpus data/processed/corpus10k.txt

# Après avoir créé un corpus de 100k caractères
python train.py --name gpt100k --corpus data/processed/corpus100k.txt

# Variante du premier essai : contexte et dimension différents
python train.py --name gpt10k_ctx128_dim64 --corpus data/processed/corpus10k.txt --block-size 128 --embedding-dim 64 --num-heads 4 --num-layers 2 --learning-rate 0.0003 --max-steps 5000 --seed 42
```

Les noms sont libres : `gpt100k` ne vérifie pas la taille du corpus. Sa taille réelle et son empreinte SHA-256 sont enregistrées. Le fichier `data/processed/corpus100k.txt` doit être préparé avant de lancer l'exemple correspondant.

Pour l'expérience `gpt10k`, les fichiers produits sont :

```text
runs/gpt10k/best_model.pt    # Meilleur modèle selon la validation
runs/gpt10k/config.json      # Corpus, réglages, graine, architecture, périphérique
runs/gpt10k/metrics.csv      # step,train_loss,val_loss
runs/gpt10k/samples.txt      # Générations intermédiaires, si activées
```

Le CSV contient une ligne à l'étape 0, puis toutes les `--eval-interval` mises à jour et à l'étape finale. Les deux pertes sont des moyennes sur `--eval-iters` lots évalués sans gradients. `step` désigne le nombre de mises à jour effectuées. Les lignes sont écrites immédiatement pour conserver les mesures si l'entraînement est interrompu. Le checkpoint inclut aussi les pertes de son étape et les métadonnées de l'expérience.

Un nom déjà utilisé est refusé pour conserver les résultats précédents. Sans `--name`, un nom horodaté en UTC est généré. `--output-dir` permet de choisir un autre dossier de résultats. `--generate-interval 0` désactive les générations intermédiaires. Les valeurs par défaut restent celles du premier entraînement, avec une graine fixée à 42 ; cette graine ne garantit pas une identité des résultats entre périphériques ou versions de PyTorch.

Générer avec le modèle de son choix :

```bash
python generate.py "il etait une fois" --tokens 1000 --checkpoint runs/gpt10k/best_model.pt
python generate.py "il etait une fois" --tokens 1000 --checkpoint runs/gpt100k/best_model.pt
```

`generate.py` demande explicitement `--checkpoint` pour choisir l’expérience à utiliser. Ces checkpoints permettent la génération ; la reprise automatique d'un entraînement n'est pas encore implémentée.

Toutes les options sont accessibles avec `python train.py --help`.

### Mesures enregistrées dans la configuration

Les nouvelles expériences ajoutent ces mesures à `runs/<nom>/config.json` et aux métadonnées du checkpoint :

| Section / champ | Signification |
| --- | --- |
| `parameter_count` | Nombre total de paramètres, conservé pour compatibilité. |
| `model_metrics` | Paramètres totaux, entraînables et gelés ; répartition par composant ; types numériques ; octets des paramètres et des buffers. |
| `training_budget` | Tokens par mise à jour, budget prévu, passages équivalents sur le corpus, tokens évalués par partition, ratio paramètres/tokens du corpus d'entraînement et perte de référence uniforme. |
| `training_metrics` | Dernière évaluation : étapes effectuées, tokens traités, passages équivalents, temps écoulé, débit global, pertes, écart validation–entraînement, perplexité, bits par caractère, meilleure perte et étape associée. |

`training_metrics` est actualisé à chaque évaluation, y compris la dernière. Dans un checkpoint, ces mesures décrivent l'évaluation à laquelle les poids ont été sauvegardés ; dans le JSON, elles décrivent la dernière évaluation effectuée. Les expériences anciennes ne sont pas réécrites.

- **Paramètres** : la configuration historique avec 33 caractères, une dimension de 128, 4 blocs, 4 têtes et un contexte de 64 comporte **807 969 paramètres**. Changer le vocabulaire modifie aussi ce nombre.
- **Mémoire** : `parameter_bytes` correspond aux poids seuls. Les buffers sont comptés séparément. Cela ne représente ni la mémoire maximale à l'entraînement (gradients, optimiseur, activations), ni la taille du checkpoint sur disque.
- **Budget** : les passages équivalents correspondent aux tokens traités divisés par la longueur de la partition d'entraînement. Les fenêtres sont tirées aléatoirement et peuvent se chevaucher : ce ne sont pas des parcours exhaustifs du corpus.
- **Qualité** : la perplexité est `exp(val_loss)` et les bits par caractère sont `val_loss / ln(2)`. Plus bas est généralement meilleur, sur le même texte de validation et avec la même tokenisation. La perplexité est `null` si son exponentielle dépasse la plage prévue ; la perte reste disponible. L'écart entre les pertes est un indicateur à suivre, pas un diagnostic de surapprentissage à lui seul.
- **Vitesse** : le débit global rapporte les tokens d'entraînement au temps écoulé dans la boucle jusqu'à l'évaluation. Il inclut les évaluations et les générations/sauvegardes déjà exécutées ; il ne mesure pas la vitesse pure du GPU et exclut l'initialisation du modèle. À l'étape zéro, le débit vaut zéro. Comparer à périphérique et fréquence d'évaluation identiques.

### Corpus synthétique à thèmes variés

Le fichier `data/processed/generaliste1m.txt` contient environ un million de caractères mêlant situations quotidiennes, dialogues, explications et exercices numériques. La [documentation du corpus](data/README_GENERALISTE.md) fournit la commande d'entraînement, sa provenance et les limites : ses structures sont générées à partir de textes de base réutilisés, et une validation aléatoire ne suffit pas à mesurer une compétence généraliste. Les corpus précédents sont conservés.

## Organisation et vérifications

Les commandes `train.py` et `generate.py` restent à la racine. Les imports passent par le paquet `my_gpt`, par exemple `from my_gpt.models.gpt import GPT`. Pour exécuter une démonstration interne, utiliser `python -m my_gpt.models.gpt` ou `python -m my_gpt.layers.transformer_block`.

Les corpus sont dans `data/processed/`. Les [sources et règles de préparation](data/README.md) restent documentées dans `data/`. Chaque expérience regroupe ses poids et ses mesures sous `runs/<nom>/`. Les anciens checkpoints ont été déplacés sans modifier leurs octets ; leurs métadonnées internes gardent les chemins historiques. Les JSON de configuration indiquent les chemins actuels et conservent les anciens dans `paths_before_reorganization_2026_10_11`.

```bash
python train.py --name segur100k_nouvel_essai --corpus data/processed/segur100k.txt --embedding-dim 64 --num-layers 2 --block-size 128
python generate.py "je pose " --checkpoint runs/segur100k_nouvel_essai/best_model.pt --temperature 0.7
python -m unittest discover -s tests -v
```
