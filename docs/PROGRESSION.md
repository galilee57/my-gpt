# Journal de progression — my-gpt

Ce journal conserve les étapes, les résultats et leurs limites pour montrer la progression du projet. Les observations concernent les sorties réellement disponibles ; les améliorations envisagées restent des expériences à réaliser.

## Parcours de construction

1. **Prédiction caractère par caractère** : création d'un tokenizer et d'un corpus initial de trois phrases, soit 153 caractères.
2. **Modèle Bigram** : première boucle d'apprentissage, avec prédiction à partir du seul caractère courant.
3. **Attention causale** : construction des projections Q, K, V et du masque empêchant de regarder le futur.
4. **Assemblage du Transformer** : plusieurs têtes, normalisation, réseau FeedForward et connexions résiduelles. Une classe par fichier pour faciliter la lecture.
5. **GPT complet** : embeddings de tokens et de positions, empilement de blocs et projection vers le vocabulaire.
6. **Entraînement et génération** : séparation entraînement/validation, sauvegarde de la configuration, du tokenizer, des poids et de l'optimiseur ; rechargement pour générer à partir d'un prompt.
7. **Premier résultat archivé, le 10 octobre 2026** : la chaîne allant du corpus à la génération fonctionne. La qualité linguistique reste à améliorer.

## Expérience 001 — Première génération, 10 octobre 2026

### Commande et sortie conservée

```bash
python generate.py "il etait une fois" --tokens 1000
```

La [sortie transmise par l'utilisateur](results/2026-10-10-premiere-generation.txt) est conservée séparément, avec ses erreurs, sa ponctuation et ses sauts de ligne. Elle constitue notre premier point de comparaison. Le nombre 1000 correspond aux nouveaux tokens demandés ; le journal ne certifie pas que la transcription fournie contient toute la sortie du terminal.

Dans ce projet, un token est un caractère. Le texte affiché inclut le prompt : les mots « il etait une fois » viennent de l'utilisateur et ne sont donc pas une réussite de génération du modèle.

### Configuration et mesures vérifiées

Les informations suivantes proviennent du checkpoint `checkpoints/best_gpt.pt` et des fichiers présents lors de l'analyse. Aucun identifiant dans la sortie fournie ne permet de prouver qu'elle provient exactement de cette version du checkpoint.

| Élément | Valeur observée |
| --- | --- |
| Corpus actuel | 10 622 caractères, environ 1 863 mots |
| Vocabulaire du checkpoint | 33 caractères |
| Partition actuelle du corpus | 9 559 tokens pour l'entraînement, 1 063 pour la validation |
| Dimension des embeddings | 128 |
| Blocs Transformer | 4 |
| Têtes par bloc | 4 |
| Fenêtre de contexte | 64 caractères |
| Paramètres entraînables | 807 969 |
| Étape enregistrée | 250 |
| Perte de validation enregistrée | 2,1868 |
| Cible du script actuel | 5 000 mises à jour |
| Batch / taux d'apprentissage actuels | 32 / 0,0003, AdamW |
| Échantillonnage actuel | Tirage multinomial sur le softmax ; température implicite de 1, sans top-k ni top-p |
| Graine aléatoire | Non fixée |

Dans la boucle actuelle, l'évaluation précède la mise à jour de l'étape correspondante : `step=250` représente 250 mises à jour déjà effectuées. Le checkpoint conserve le meilleur résultat de validation rencontré aux évaluations, pas nécessairement la dernière étape exécutée. Sa présence ne dit pas si l'entraînement a été arrêté ou poursuivi.

La perte est une entropie croisée moyenne par caractère, exprimée en nats. À titre de repère mathématique, une distribution uniforme sur 33 caractères donne une perte de `ln(33) ≈ 3,4965`. La valeur enregistrée est inférieure à ce repère : le modèle prédit les caractères mieux qu'une distribution uniforme sur les lots évalués. Cela ne mesure pas directement la qualité des phrases. La perte d'entraînement et la courbe des pertes ne sont pas disponibles pour cette expérience ; on ne peut pas établir un surapprentissage ou une convergence à partir de cette seule valeur.

### Lecture du résultat

| Observation | Interprétation et limite |
| --- | --- |
| Mots reconnaissables : « chat », «les », « pas », « de » | Le texte contient des unités plausibles, mais leur présence seule ne prouve pas la maîtrise du vocabulaire. |
| Espaces, points et retours à la ligne | La sortie ressemble visuellement à du texte et contient des séparations ; la ponctuation reste parfois mal placée. |
| Formes comme « taavauxend », « conimavac », « strqua » | Les mots sont souvent déformés ; l'orthographe reste instable. |
| Répétition « de ge da la la » | Les choix locaux peuvent produire des répétitions sans structure de phrase. |
| Absence d'histoire suivie après le prompt | Cet échantillon ne montre pas de cohérence narrative ni de continuité des personnages ou des actions. |
| Absence de séparation après « fois » | Le modèle continue directement le dernier mot. Un prompt terminé par un espace peut être testé séparément. |

Le résultat est compatible avec un apprentissage encore limité des régularités locales. Il ne suffit pas à identifier une cause unique : quantité et diversité du corpus, nombre de mises à jour, contexte disponible et caractère aléatoire de la génération peuvent tous intervenir.

Le corpus actuel est petit face aux 807 969 paramètres du modèle. Il contient surtout des phrases courtes et quelques histoires. Le prompt « il etait une fois » n'y figure pas textuellement. La fenêtre de 64 caractères limite aussi la quantité de texte visible à chaque prédiction : demander 1000 caractères ne porte pas cette fenêtre à 1000.

### Analyse complémentaire de ChatGPT — retour fourni le 10 octobre 2026

L'utilisateur a transmis une analyse de ChatGPT de cette première génération. Elle souligne le jalon technique atteint : **le modèle a été sauvegardé, rechargé et utilisé pour générer du texte à partir d'un prompt**, avec une demande de 1 000 nouveaux caractères. Cette validation du fonctionnement constitue une étape distincte de la qualité linguistique obtenue.

ChatGPT s'appuie notamment sur ces extraits de la sortie :

```text
il etait une foise taavauxend chat aou.
lll feret'etoivent che dercuele voanct vetnchenur senodens
llebl s va farede cinte pas, tent ain dest ce lenonge.
```

Son interprétation est que le GPT commence à apprendre certaines régularités du français :

- Une alternance de voyelles et de consonnes qui paraît parfois naturelle.
- Des fragments ressemblant à des mots français.
- L'utilisation d'espaces, d'apostrophes et de ponctuation.
- Des terminaisons évoquant des formes fréquentes de la langue.

Ces observations suggèrent un début d'apprentissage des régularités locales. Elles restent qualitatives : aucune mesure de l'alternance des lettres ou de la fréquence des terminaisons n'a été réalisée. La construction des mots et la syntaxe ne sont pas encore maîtrisées dans cet échantillon. Le prompt fourni doit toujours être distingué de la continuation produite par le modèle.

**Hypothèse mise en avant : un corpus trop petit.** ChatGPT propose de vérifier d'abord la quantité de texte disponible et suggère un corpus d'au moins **100 000 caractères**, idéalement de plusieurs millions, pour expérimenter. Comparé aux 10 622 caractères du corpus actuel, le premier palier représente environ **9,4 fois plus de texte**.

Ce chiffre est conservé comme une proposition d'expérience issue du retour de ChatGPT, et non comme un seuil universel garantissant des phrases correctes. L'effet devra être évalué avec des textes variés, une validation séparée et un suivi de l'entraînement. L'échantillon seul ne permet pas d'attribuer toutes les difficultés au corpus.

**Expérience à envisager :** constituer une deuxième version du corpus proche de 100 000 caractères, conserver la version initiale et ses résultats, puis comparer les pertes et plusieurs générations sur les prompts de référence. Consigner le budget d'entraînement : avec davantage de données, un même nombre de mises à jour ne représente pas le même nombre de passages sur le corpus.

L'extrait transmis annonce trois explications possibles, mais s'interrompt après la partie « A. Un corpus trop petit ». Les deux autres explications n'ont pas été fournies et ne sont pas reconstituées ici. Cette intégration documentaire ne modifie ni le corpus ni le modèle et ne lance pas de nouvel entraînement.

### Avertissement NumPy

Le terminal signale `Failed to initialize NumPy: No module named 'numpy'`. La génération s'est néanmoins exécutée. Les chemins utilisés ici passent par des tenseurs PyTorch et des listes Python ; cet avertissement n'explique pas, à lui seul, la mauvaise qualité linguistique. Il constitue un point d'environnement à traiter séparément de l'analyse du modèle.

### Éléments à améliorer et protocole de comparaison

1. **Tracer l'entraînement** : conserver les pertes train/validation, l'étape du checkpoint, le temps écoulé et le périphérique. Observer leur évolution avant de décider de prolonger l'apprentissage.
2. **Comparer plusieurs sorties** : utiliser les mêmes prompts, une même longueur et des graines enregistrées, avec plusieurs tirages par configuration. Une seule génération peut être trompeuse.
3. **Tester le corpus** : ajouter des textes suivis et variés, puis comparer à réglages constants. Garder des textes de validation séparés ; une répétition du corpus augmente surtout le risque de mémorisation.
4. **Tester l'échantillonnage** : ajouter ensuite une température configurable et éventuellement top-k. Cela peut améliorer la lisibilité, sans remplacer l'apprentissage.
5. **Tester le contexte** : comparer une fenêtre plus longue quand les mots et les phrases deviennent plus stables, en tenant compte du coût de calcul.

Prompts de référence pour les expériences suivantes :

```text
il etait une fois
le chat
bonjour, comment
si tu as soif,
```

Utiliser par exemple 200 nouveaux caractères par prompt pour faciliter la lecture côte à côte. Tester aussi `"il etait une fois "` avec un espace final, en conservant ce changement dans les notes. Le script actuel ne propose pas encore d'option de graine ou de température ; ces réglages devront être ajoutés pour mieux contrôler les comparaisons.

### Traçabilité des fichiers analysés

Le checkpoint ne contient pas l'empreinte du corpus d'entraînement. Les empreintes ci-dessous identifient les fichiers présents au moment de cette analyse ; elles ne prouvent pas rétroactivement quel corpus a servi à entraîner les poids.

```text
SHA-256 data/corpus.txt
 e4cfac3aa0c5c97401c61fbbe62d17caefdecbd895cf9ad0a79abea2c0d21cb4
SHA-256 checkpoints/best_gpt.pt
 1a032f668cf1f406d6793c852e74606559da4d3883c11e82d6583e1e90645678
```

`best_gpt.pt` est écrasé lorsqu'un meilleur checkpoint est sauvegardé. Pour une future expérience, conserver une copie portant son identifiant avant de relancer l'entraînement, ainsi que le corpus et les logs correspondants.

## Fiche pour les prochaines expériences

Copier cette fiche et ajouter une nouvelle entrée, sans remplacer les résultats précédents.

| Champ | À renseigner |
| --- | --- |
| Identifiant / date | |
| Hypothèse testée | |
| Changement par rapport à la référence | |
| Corpus : taille, provenance, empreinte, séparation | |
| Architecture et nombre de paramètres | |
| Batch, taux d'apprentissage, nombre de mises à jour | |
| Pertes train/validation et étape du checkpoint | |
| Checkpoint : chemin conservé et empreinte | |
| Périphérique, versions et durée | |
| Prompts, graines, longueur et échantillonnage | |
| Sorties archivées | |
| Orthographe, syntaxe, cohérence, répétitions | |
| Conclusion et prochaine expérience | |

## Suivi des expériences nommées — 10 octobre 2026

L'entraînement accepte désormais un nom, un corpus et des hyperparamètres en ligne de commande. Chaque expérience conserve son meilleur checkpoint `checkpoints/best_<nom>.pt`, ses mesures `runs/<nom>/metrics.csv` et sa configuration `runs/<nom>/config.json`. Les générations intermédiaires sont également archivées quand elles sont activées.

Le CSV associe `step`, `train_loss` et `val_loss` à chaque évaluation, y compris avant la première mise à jour et après la dernière. Les noms existants sont refusés pour éviter de remplacer une expérience. La configuration enregistre notamment la graine, la taille et l'empreinte du corpus, le périphérique et la version de PyTorch. Ces ajouts s'appliquent aux nouveaux essais ; les métadonnées manquantes du premier résultat ne sont pas reconstituées rétroactivement.

Validation technique : deux entraînements courts sous des noms différents, évaluation finale hors intervalle, correspondance entre le meilleur checkpoint et le CSV, génération depuis ces checkpoints et refus d'écraser un essai existant. Ces tests vérifient le suivi des expériences et ne constituent pas de nouveaux résultats de qualité linguistique.

Voir les [commandes dans le README](../README.md#comparer-plusieurs-expériences). La graine d'entraînement est maintenant configurable ; celle de la génération autonome et la température restent à ajouter.

## Réorganisation du projet — 11 octobre 2026

Le code réutilisable est regroupé dans le paquet `my_gpt` : tokenizer et dataset, modèles dans `models/`, composants du Transformer dans `layers/`. Les commandes `train.py` et `generate.py` restent à la racine. Les corpus et leurs métadonnées sont maintenant dans `data/processed/` ; `data/raw/` est réservé aux futurs textes sources.

Chaque expérience regroupe désormais son checkpoint et ses mesures dans `runs/<nom>/`. Par exemple, `checkpoints/best_segur100k.pt` devient `runs/segur100k/best_model.pt`. Les poids et les corpus ont été conservés octet pour octet. Les fichiers `config.json` utilisent les nouveaux chemins et enregistrent les anciens dans `paths_before_reorganization_2026_10_11`. Les métadonnées embarquées dans les checkpoints restent historiques pour préserver les fichiers originaux. Les chemins cités dans les expériences précédentes de ce journal décrivent donc l'organisation de l'époque.

Les ajouts manuels de température à `gpt.py` et `generate.py` sont conservés. La génération demande maintenant explicitement `--checkpoint`, pour choisir l'expérience souhaitée. Le corpus par défaut de l'entraînement est `data/processed/corpus10k.txt`.

```bash
python generate.py "je pose " --checkpoint runs/segur100k_dim64_layers2_ctx128/best_model.pt --temperature 0.7
python -m unittest discover -s tests -v
```

Validation : tests des gradients et du masque causal, entraînement court jusqu'à l'évaluation finale, sauvegarde et rechargement, protection contre l'écrasement d'une expérience, génération avec les sept checkpoints existants, vérification des empreintes des corpus et des checkpoints déplacés. Aucun entraînement complet supplémentaire n'a été lancé.
