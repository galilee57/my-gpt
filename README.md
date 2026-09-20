# my-gpt
LLM from scratch with pytorch   

## Au départ, on constitue un corpus simple.
Le LLM va être entrainé pour prédire la lettre suivante.
Les LLM ne fonctionnenent pas pas de cette façon, mais pédagogiquement, c'est la première étape à implémenter.

## 1ère étape : Tokenizer (tokenizer.py)
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

## 2ème étape : constitution du dataset pour l'entrainement (dataset.py)
Le GPT ne reçoit pas tout le texte : il faut lui définir une fenêtre `block_size=8` que le modèle regarde au maximum.  
On construit `x=chunk[:-1] et y=chunk[1:]` ce qui donne `x : b o n j o u r - y : o n j o u r`.
Par ce décalage, on crée en une séance plusieurs exercices : le modèle apprends simultanéménet b-> o, bo -> n, ...
C'est l'idée de **l'entrainement autoregressif** : avec un block_size=8 et batch_size=4, on aura 4 séances de 8 tokens.

## 3ème étape : construction du premier modèle BIGRAM (model.py)
BAsé sur Bayes-Naive : P(xt=1|xt).
On crée une couche neuronale (nn.embedding) : si par ex vocab_size=30, PyTorch crée une matrice 30x30 : chaque token aura un vecteur de 30 nombres.
Le modèle renvoie des LOGITS (score), qui sont ensuite transformés via SOFTMAX.
La CROSSENTROPY permet de mesurer la distance entre prédiction et bonne réponse, puis on calcule le LOSS que l'on cherche à minimiser.

## 4ème étape : encodage du TRANSFORMER


