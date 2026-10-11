"""Construire un corpus pédagogique synthétique, diversifié mais à structures contrôlées.

Aucun texte téléchargé, aucune API et aucun remplissage par copie de paragraphes.
Les connaissances et situations sont rédigées dans ce script ; les exercices sont calculés.
"""
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261011
TARGET = 1_000_000

# sujet | explication | exemple | confusion à éviter | question de réflexion
LESSONS_TEXT = '''
cycle de l'eau|l'eau peut s'evaporer, se condenser puis retomber sous forme de precipitation. ces transformations font circuler l'eau entre l'air, les sols et les etendues d'eau.|des gouttelettes apparaissent sur une paroi froide quand de la vapeur d'eau se condense.|un nuage est constitue de gouttelettes ou de cristaux, pas simplement de vapeur invisible.|quelles transformations permettent a l'eau de circuler ?
photosynthese|les plantes utilisent l'energie lumineuse pour fabriquer des substances organiques a partir d'eau et de dioxyde de carbone. ce processus libere de l'oxygene.|une plante placee dans des conditions adaptees utilise la lumiere recue par ses feuilles.|la lumiere apporte de l'energie, mais ne remplace ni l'eau ni les elements mineraux.|pourquoi la lumiere ne suffit-elle pas a elle seule ?
biodiversite|la biodiversite comprend la diversite des especes, des individus et des milieux. les interactions entre organismes contribuent au fonctionnement des ecosystemes.|une haie offre des abris et des ressources a plusieurs animaux.|compter les arbres ne suffit pas a decrire toute la diversite d'un milieu.|quelles observations completeraient un simple comptage ?
chaine alimentaire|les organismes trouvent leur energie dans des ressources alimentaires. plusieurs relations alimentaires se croisent et forment un reseau.|une chenille mange une feuille et peut devenir la proie d'un oiseau.|un reseau alimentaire ne se reduit pas toujours a une seule suite lineaire.|pourquoi plusieurs especes peuvent-elles dependre d'une meme ressource ?
etats de la matiere|un corps peut etre solide, liquide ou gazeux selon les conditions. un changement d'etat ne signifie pas necessairement un changement de substance.|de la glace qui fond devient de l'eau liquide.|fondre et se dissoudre sont deux phenomenes differents.|quelle difference existe entre la fonte d'un glacon et la dissolution du sel ?
dissolution|lorsqu'une substance se dissout dans un liquide, ses constituants se dispersent dans ce liquide. la quantite pouvant se dissoudre depend des substances et des conditions.|du sel ajoute a de l'eau peut former une solution transparente.|une substance dissoute n'a pas simplement cesse d'exister.|comment distinguer une disparition visuelle et une absence de matiere ?
force et mouvement|une force peut modifier le mouvement ou deformer un objet. plusieurs forces peuvent agir en meme temps et leurs effets dependent de leur combinaison.|une main pousse un chariot tandis que des frottements s'opposent a son mouvement.|un objet en mouvement n'a pas toujours besoin d'une force dans le sens de son mouvement pour continuer.|quelles forces faut-il examiner avant de conclure ?
son|le son correspond a une vibration qui se propage dans un milieu materiel. la hauteur d'un son depend de sa frequence.|une corde qui vibre peut mettre l'air voisin en vibration.|la hauteur et le volume sonore ne decrivent pas la meme propriete.|comment un son peut-il etre aigu sans etre fort ?
lumiere et ombre|une ombre se forme lorsqu'un objet bloque une partie de la lumiere. sa forme depend de la source, de l'objet et de la surface qui la recoit.|l'ombre d'un arbre change pendant la journee lorsque la position apparente du soleil change.|une ombre ne donne pas toujours les dimensions reelles de l'objet.|pourquoi deux ombres d'un meme objet peuvent-elles etre differentes ?
energie|l'energie peut etre transferee ou convertie d'une forme a une autre. une partie devient souvent moins facile a utiliser, par exemple sous forme de chaleur diffuse.|une lampe transforme de l'energie electrique en lumiere et en chaleur.|dire qu'un appareil consomme de l'energie ne veut pas dire que cette energie est detruite.|quelles transformations observe-t-on autour d'un appareil ?
isolation thermique|un isolant ralentit les transferts de chaleur. son effet depend aussi de son epaisseur, de sa pose et des autres parties du batiment.|un manteau ralentit les pertes de chaleur du corps.|un isolant ne produit pas lui-meme une source permanente de chaleur.|pourquoi une mauvaise fermeture peut-elle reduire l'effet d'un isolant ?
meteo et climat|la meteo decrit les conditions atmospheriques sur des durees courtes. le climat repose sur des observations et des statistiques portant sur des durees longues.|une journee froide peut survenir dans une periode dont la temperature moyenne augmente.|une seule journee ne permet pas de caracteriser une tendance climatique.|quel ensemble d'observations faudrait-il comparer ?
rotation de la terre|la rotation de la terre explique l'alternance habituelle du jour et de la nuit. a un instant donne, seule une partie du globe est eclairee par le soleil.|des lieux eloignes peuvent connaitre des heures differentes.|le jour et la nuit ne sont pas causes par la terre qui cache chaque soir le soleil a tout le monde simultanement.|pourquoi deux personnes peuvent-elles observer des moments differents de la journee ?
carte et echelle|l'echelle relie une distance sur une carte a une distance sur le terrain. il faut conserver des unites coherentes pour faire le calcul.|une carte a l'echelle 1 pour 1000 represente un metre sur le terrain par un millimetre sur la carte.|une echelle n'est pas une indication de duree du trajet.|quelles informations manquent pour estimer un temps de marche ?
relief et cours d'eau|le relief influence les chemins suivis par l'eau. un bassin versant rassemble les surfaces dont les eaux s'ecoulent vers une meme sortie.|la pluie tombee sur deux versants peut rejoindre des rivieres differentes.|une frontiere administrative ne correspond pas toujours a une limite naturelle.|pourquoi le relief compte-t-il dans la circulation de l'eau ?
chronologie|une chronologie classe des evenements dans le temps. elle aide a distinguer ce qui precede, ce qui suit et ce qui se produit pendant une meme periode.|une frise peut situer la construction d'un pont avant sa renovation.|le fait qu'un evenement en precede un autre ne prouve pas qu'il en soit la cause.|que faut-il chercher en plus de l'ordre des dates ?
sources historiques|une source historique doit etre examinee en tenant compte de son auteur, de sa date et de son contexte. plusieurs sources peuvent apporter des points de vue differents.|un journal intime et un registre officiel ne racontent pas un evenement de la meme maniere.|un document ancien n'est pas automatiquement complet ou impartial.|quelles questions poser avant d'utiliser un temoignage ?
patrimoine|le patrimoine peut comprendre des batiments, des objets, des pratiques et des savoir-faire transmis. sa conservation implique des choix et des ressources.|un atelier peut transmettre une technique de fabrication a de nouveaux artisans.|conserver un objet ne suffit pas toujours a conserver le savoir permettant de l'utiliser.|quels gestes et quelles explications faudrait-il transmettre ?
langue et registre|le registre de langue depend de la situation et de la relation entre les personnes. on peut exprimer une meme idee avec des formulations differentes.|une demande a un ami peut etre reformulee de facon plus formelle dans un courrier.|une formulation familiere n'est pas adaptee a toutes les situations.|quels indices permettent de choisir le ton d'un message ?
phrase et ponctuation|la ponctuation organise les relations entre les parties d'un texte. elle aide a distinguer une question, une affirmation ou les elements d'une enumeration.|un point termine une phrase et un point d'interrogation signale une question directe.|une virgule ne peut pas remplacer tous les autres signes sans changer la lecture.|comment rendre une phrase longue plus facile a suivre ?
recit|un recit relie des evenements, des personnages et des lieux. les liens entre les actions aident le lecteur a comprendre le deroulement de l'histoire.|un personnage perd un objet, le cherche puis explique comment il l'a retrouve.|une succession de phrases ne forme pas automatiquement une histoire coherente.|quel lien manque entre deux actions apparemment separees ?
argumentation|une argumentation presente une idee et les raisons qui la soutiennent. un exemple illustre un argument, mais ne suffit pas toujours a etablir une regle generale.|on peut discuter l'ouverture d'une salle en comparant les besoins, les horaires et les moyens.|repeter une opinion ne lui apporte pas de preuve supplementaire.|quelles observations pourraient contredire une proposition ?
lecture critique|lire de facon critique consiste a distinguer ce qui est affirme, ce qui est montre et ce qui reste incertain. la provenance du document aide a interpreter son contenu.|une affiche peut vouloir convaincre tandis qu'un compte rendu cherche a decrire une reunion.|une presentation soignee ne garantit pas l'exactitude de toutes les affirmations.|comment verifier une information sans se fier uniquement a sa forme ?
musique|le rythme organise les durees et les accents. la melodie correspond a une succession de hauteurs, tandis que le timbre aide a distinguer les sources sonores.|deux instruments peuvent jouer une meme melodie avec des timbres differents.|jouer plus vite et jouer plus fort ne sont pas la meme chose.|quels elements changent dans deux interpretations d'un meme morceau ?
photographie|une photographie depend du cadrage, de la lumiere et du moment choisi. ce qui reste hors du cadre peut modifier notre interpretation de la scene.|un plan rapproche montre un detail mais peut masquer son environnement.|une image ne donne pas toujours tout le contexte de ce qu'elle represente.|quelles informations ne sont pas visibles dans le cadre ?
proportionnalite|deux quantites sont proportionnelles lorsque l'une s'obtient en multipliant l'autre par un coefficient constant. ce lien doit etre verifie dans la situation etudiee.|a prix unitaire fixe, doubler le nombre d'objets double le prix total.|toutes les grandeurs qui augmentent ensemble ne sont pas proportionnelles.|comment verifier si le coefficient reste constant ?
moyenne et dispersion|la moyenne resume un ensemble de valeurs par un nombre. la dispersion indique a quel point ces valeurs sont eloignees les unes des autres.|deux groupes peuvent avoir la meme moyenne mais des resultats tres differents entre leurs membres.|la moyenne ne decrit pas a elle seule toutes les valeurs d'un groupe.|quelles informations ajouter pour mieux decrire les differences ?
probabilite|une probabilite decrit la possibilite d'un evenement dans un modele donne. elle ne garantit pas le resultat d'un essai particulier.|un tirage aleatoire peut donner plusieurs fois de suite le meme resultat.|un resultat rare n'est pas pour autant impossible.|quelle difference existe entre une prevision et une certitude ?
correlation|une correlation decrit une association entre des variations observees. cette association peut avoir plusieurs explications, dont une cause commune.|deux mesures peuvent augmenter pendant une meme saison sans que l'une cause directement l'autre.|observer une association ne suffit pas a demontrer un lien causal.|quelle autre variable pourrait influencer les deux mesures ?
mesure et precision|une mesure depend d'un instrument et d'une methode. indiquer l'unite et la precision utile aide a interpreter le resultat.|une regle graduee au millimetre ne justifie pas toutes les decimales affichees par un calcul.|un nombre tres detaille n'est pas toujours une mesure tres exacte.|quelles limites de l'instrument faut-il signaler ?
algorithme|un algorithme decrit une suite d'etapes pour accomplir une tache. les cas limites doivent etre precises pour eviter des resultats inattendus.|une recherche dans une liste doit prevoir le cas ou l'element demande est absent.|une instruction ambigue peut etre comprise par une personne sans etre executable telle quelle par un programme.|comment rendre une etape assez precise pour etre repetee ?
variable informatique|une variable permet de designer une valeur dans un programme. sa signification depend du contexte et des operations realisees.|un compteur peut etre augmente apres chaque element traite.|le nom d'une variable ne remplace pas la verification de sa valeur.|quelles valeurs faut-il regarder lorsque le resultat semble faux ?
fichier et sauvegarde|un fichier conserve des donnees dans un format. une sauvegarde permet de retrouver une version si l'original est perdu ou endommage.|conserver une copie sur un autre support reduit certains risques de perte.|une copie dans le meme dossier ne protege pas contre toutes les pannes du support.|comment verifier qu'une sauvegarde peut etre relue ?
tableau de donnees|un tableau organise souvent les observations en lignes et les proprietes en colonnes. des noms de colonnes et des unites explicites evitent des confusions.|une colonne de distances doit indiquer si les valeurs sont en metres ou en kilometres.|une cellule vide ne signifie pas toujours zero.|comment distinguer une valeur manquante et une quantite nulle ?
modele de langage|un modele de langage estime des probabilites de tokens a partir d'un contexte. l'entrainement ajuste ses parametres en comparant des predictions a des exemples.|un modele caractere peut essayer de predire la lettre suivante d'un mot.|un texte fluide ne garantit pas que les faits qu'il contient soient exacts.|pourquoi faut-il verifier une affirmation meme si elle est bien formulee ?
validation d'un modele|les exemples de validation servent a mesurer un comportement sur des donnees reservees. ils doivent rester distincts des exemples utilises pour ajuster les parametres.|une perte d'entrainement qui baisse tandis que celle de validation monte peut indiquer un surapprentissage.|un bon score sur les exemples memorises ne suffit pas a montrer une generalisation.|quelles donnees faut-il tenir a l'ecart pendant l'apprentissage ?
comparaison d'experiences|une comparaison est plus interpretable quand les changements et les conditions sont notes. plusieurs essais peuvent aider a distinguer un effet stable d'une variation aleatoire.|on peut garder un meme jeu de validation et ne modifier qu'un hyperparametre.|changer en meme temps le corpus, le modele et le test complique l'attribution d'un progres.|quel element faut-il maintenir constant pour tester une hypothese ?
organisation du travail|une tache devient plus facile a suivre lorsqu'on distingue l'objectif, les etapes et les contraintes. des points de controle permettent d'ajuster le travail.|une equipe peut verifier une premiere version avant de terminer tous les details.|une liste tres longue n'indique pas automatiquement ce qui est prioritaire.|quelle etape reduirait le plus l'incertitude actuelle ?
cooperation|cooperer suppose de partager des informations et de clarifier les responsabilites. des attentes explicites reduisent les malentendus.|deux personnes peuvent convenir de qui apporte le materiel et de qui prepare la salle.|le silence d'un partenaire ne prouve pas qu'il a accepte une proposition.|quelle confirmation faut-il obtenir avant de repartir les taches ?
consommation et dechets|reduire les dechets peut passer par l'entretien, la reutilisation et des choix adaptes aux besoins. les possibilites dependent des objets et des filieres disponibles.|reparer une fermeture peut prolonger l'usage d'un sac.|tous les materiaux ne se trient pas de la meme maniere dans tous les endroits.|quelles consignes locales faut-il consulter avant de trier ?
jardinage|le developpement d'une plante depend notamment de l'eau, de la lumiere, du sol et de l'espece. observer les conditions aide a choisir un entretien adapte.|deux pots places dans des endroits differents peuvent secher a des vitesses differentes.|arroser davantage ne resout pas tous les problemes d'une plante.|quelles observations faire avant de changer l'arrosage ?
recette|une recette organise des ingredients, des quantites et des etapes. certaines transformations dependent aussi de la temperature, du temps et du materiel.|on peut adapter des quantites pour un autre nombre de portions, puis verifier le resultat.|multiplier les ingredients ne signifie pas toujours multiplier le temps de cuisson dans la meme proportion.|quelles parties de la preparation demandent une verification plutot qu'une simple multiplication ?
transport|un trajet depend de la distance, du mode de deplacement et des conditions. une duree estimee doit tenir compte des arrets et des correspondances.|un parcours court peut prendre plus de temps si une correspondance impose une attente.|la distance seule ne suffit pas a determiner l'heure d'arrivee.|quelles etapes composent le temps total du trajet ?
accessibilite|l'accessibilite vise a permettre l'usage d'un lieu ou d'un service par des personnes ayant des besoins differents. elle concerne aussi l'information et les outils.|un document lisible et une entree sans obstacle peuvent faciliter la participation.|une solution adaptee a une personne ne couvre pas automatiquement tous les besoins.|qui faut-il consulter pour identifier les obstacles reels ?
observation et hypothese|une observation decrit ce qui a ete constate. une hypothese propose une explication qui peut etre examinee par de nouvelles observations.|on constate une fuite puis on cherche si elle vient du joint ou d'une autre piece.|nommer une cause possible ne prouve pas que cette cause est la bonne.|quelle verification permettrait de departager deux explications ?
experience scientifique|une experience cherche a examiner une hypothese dans des conditions decrites. des repetitions et des controles aident a interpreter les differences.|on peut comparer des graines placees dans des conditions similaires sauf pour un facteur etudie.|une difference observee une seule fois peut avoir plusieurs explications.|quelles conditions faut-il rendre comparables ?
resume|un resume conserve les idees principales d'un texte en reduisant les details. il doit respecter le sens et distinguer les affirmations des incertitudes.|un compte rendu court peut reprendre une decision, sa raison et la prochaine etape.|resumer ne consiste pas a ajouter des conclusions absentes du texte.|quelles informations sont indispensables a la comprehension ?
definition|une definition explique le sens d'un terme dans un contexte. un exemple peut aider, mais il ne remplace pas toujours les proprietes qui distinguent ce terme.|on peut definir un triangle comme un polygone a trois cotes, puis en dessiner plusieurs.|un seul exemple ne montre pas toutes les formes possibles d'une categorie.|quelles proprietes restent vraies pour tous les exemples de la categorie ?
accord et desaccord|un desaccord peut porter sur les faits, les objectifs ou les preferences. reformuler le point discute aide a chercher une solution.|deux personnes peuvent vouloir une salle calme mais proposer des horaires differents.|ne pas partager une preference ne signifie pas manquer de respect.|sur quel objectif commun les personnes peuvent-elles s'appuyer ?
incertitude|exprimer une incertitude consiste a preciser ce qui est connu et ce qui reste a verifier. cela aide a choisir une prochaine action utile.|un temoignage incomplet peut permettre une hypothese sans permettre une conclusion definitive.|une reponse precise dans sa forme peut reposer sur des informations insuffisantes.|quelle information changerait le plus la conclusion ?
'''

# lieu | activité | objet | difficulté | première action | vérification | résultat
SCENARIOS_TEXT = '''
a la bibliotheque|preparer une selection de livres|le catalogue|plusieurs titres portent des noms proches|comparer les auteurs et les descriptions|verifier que les livres choisis correspondent au sujet|la selection devient plus facile a expliquer aux lecteurs
au jardin partage|organiser l'arrosage|le tableau des tours|deux personnes pensent intervenir le meme jour|relire ensemble les disponibilites|demander a chacun de confirmer son tour|les plantes recoivent un suivi plus regulier
au bureau|preparer une reunion|l'ordre du jour|les sujets sont trop nombreux pour le temps disponible|classer les points selon leur importance|verifier les decisions qui doivent etre prises|la reunion commence avec un objectif plus clair
au musee|preparer une visite|le plan des salles|une salle prevue est temporairement fermee|revoir le parcours avec les informations de l'accueil|verifier les acces et les temps de marche|le groupe suit un parcours adapte
au marche|organiser les achats|la liste des produits|un ingredient prevu n'est pas disponible|chercher un remplacement adapte a la recette|comparer les quantites et les usages possibles|le repas peut etre prepare sans achat inutile
au centre culturel|installer une exposition|les etiquettes|plusieurs descriptions sont difficiles a lire|reformuler les textes et revoir leur presentation|faire relire les etiquettes a une autre personne|les visiteurs comprennent mieux les informations
au club de musique|preparer une repetition|les partitions|les participants utilisent des versions differentes|comparer les references des documents|verifier que tous suivent la meme version|le groupe peut reprendre au meme endroit
au parc|preparer une promenade|la carte des chemins|le chemin habituel est bloque|chercher un autre parcours indique|verifier la distance et les conditions du nouveau trajet|la promenade peut se poursuivre sans improviser un passage dangereux
au cafe associatif|accueillir des visiteurs|le panneau d'information|les horaires affiches sont ambigus|ecrire les heures de debut et de fin clairement|faire confirmer les horaires par les organisateurs|les visiteurs savent quand revenir
au club de lecture|discuter un roman|les notes de lecture|les participants confondent deux personnages|revenir aux passages concernes|distinguer les actions de chaque personnage|la discussion s'appuie sur des exemples precis
au jardin botanique|observer des plantes|le carnet d'observation|des feuilles se ressemblent beaucoup|noter plusieurs caracteristiques visibles|comparer la forme des tiges et la disposition des feuilles|les observations deviennent plus precises sans conclure trop vite
au studio de dessin|organiser le materiel|les boites de crayons|les couleurs sont melangees et difficiles a retrouver|regrouper les crayons selon des categories simples|essayer le rangement pendant une seance|le materiel devient plus facile a partager
au theatre amateur|preparer une scene|le texte de la piece|un changement de decor prend trop de temps|repartir les gestes entre les participants|repeter la transition sans jouer toute la scene|le changement de decor devient plus fluide
au local de quartier|organiser un atelier|le formulaire d'inscription|plusieurs personnes n'ont pas indique leur disponibilite|demander les informations manquantes|verifier le nombre de places pour chaque horaire|les participants recoivent une proposition adaptee
au centre de loisirs|preparer un jeu|les regles imprimees|une consigne peut etre comprise de deux manieres|reformuler la consigne avec un exemple|faire expliquer la regle par un participant|le groupe peut jouer avec les memes attentes
au magasin de fournitures|choisir du papier|la fiche des formats|les dimensions sont indiquees dans des unites differentes|convertir les mesures dans une meme unite|comparer les dimensions utiles et non seulement les noms|le format choisi correspond au travail prevu
au club photo|comparer des images|la serie de photographies|le sujet principal est peu visible|examiner le cadrage et la lumiere|comparer plusieurs versions de la meme scene|la discussion distingue les choix techniques des preferences
au laboratoire pedagogique|organiser une observation|la fiche de mesures|une mesure semble differente des autres|verifier la methode et l'instrument|recommencer la mesure dans les memes conditions|le groupe dispose d'informations plus fiables
au bureau d'accueil|orienter un visiteur|le plan du batiment|la personne ne connait pas le nom du service cherche|lui demander de decrire sa demande|confirmer la destination avant d'indiquer le chemin|le visiteur recoit une indication plus utile
au club de sport|organiser une rencontre|le planning|deux groupes demandent le meme creneau|examiner les contraintes de chaque groupe|faire confirmer la nouvelle repartition|les participants connaissent leur horaire
au village|preparer une fete|la liste du materiel|un objet necessaire figure deux fois et un autre manque|comparer la liste avec les activites prevues|controler chaque poste avant les achats|la preparation evite plusieurs oublis
au bord de la riviere|observer un paysage|le carnet de terrain|la premiere description reste trop vague|distinguer ce qui est vu de ce qui est suppose|noter des details sur les formes et les positions|le compte rendu devient plus facile a comprendre
au poste de travail|classer des documents|les dossiers|plusieurs fichiers ont le meme nom|ajouter des noms qui indiquent le sujet et la version|ouvrir quelques fichiers pour verifier le classement|les documents utiles se retrouvent plus rapidement
au local informatique|preparer une demonstration|le programme|un exemple ne donne pas le resultat attendu|reduire le probleme a un cas simple|comparer le resultat obtenu au resultat attendu|la recherche de l'erreur devient plus ciblee
au centre de formation|preparer une explication|le support de cours|un exemple demande des connaissances non presentees|choisir un exemple plus accessible|demander a un lecteur de reformuler l'idee|l'explication devient plus progressive
au magasin de seconde main|organiser les objets|les fiches de description|certaines fiches oublient les dimensions|mesurer les objets et completer les fiches|verifier les unites et les valeurs saisies|les visiteurs peuvent mieux comparer les objets
au jardin de la maison|preparer des semis|les etiquettes des pots|plusieurs pots ne sont pas identifies|noter les informations encore disponibles|separer les identifications certaines des suppositions|le suivi des plantations devient plus clair
au coin cuisine|preparer une recette|la liste des ingredients|la recette ne correspond pas au nombre de convives|recalculer les quantites de base|verifier les etapes qui demandent une adaptation particuliere|la preparation correspond mieux au repas prevu
au point d'information|expliquer un trajet|la carte du quartier|le visiteur confond deux rues voisines|repeter le trajet avec des reperes visibles|demander au visiteur de reformuler la direction|la personne repart avec un parcours plus clair
au club de sciences|comparer deux essais|le tableau des resultats|plusieurs facteurs ont change en meme temps|identifier les differences entre les essais|preparer une comparaison avec un seul changement principal|les resultats deviennent plus faciles a interpreter
au local de traduction|relire un texte|le brouillon|un mot change de sens selon le contexte|examiner la phrase et le passage qui l'entoure|comparer plusieurs reformulations possibles|la traduction respecte mieux l'idee de depart
au bureau de projet|organiser une livraison|le calendrier|une etape depend d'une confirmation encore absente|contacter la personne qui doit confirmer|distinguer les dates certaines des dates provisoires|le calendrier rend les dependances plus visibles
au centre de documentation|chercher une information|la liste des sources|deux documents semblent se contredire|comparer leurs dates et leurs definitions|rechercher les differences de contexte|la synthese signale ce qui reste incertain
au club de cinema|preparer une discussion|la fiche du film|un avis personnel est presente comme un fait|separer la description de la scene et son interpretation|demander sur quel passage repose chaque argument|les participants peuvent discuter sans confondre leurs points de vue
au petit atelier|preparer une maquette|le dessin des pieces|les mesures du dessin ne correspondent pas au materiel|relever les dimensions disponibles|verifier les proportions avant l'assemblage|la maquette peut etre construite avec moins d'ajustements
au bureau des inscriptions|preparer un accueil|la liste des participants|une meme personne apparait plusieurs fois|comparer les informations sans fusionner des personnes differentes|confirmer les cas ambigus avec les interesses|la liste devient plus fiable
au club d'ecriture|relire une histoire|le premier chapitre|une action arrive sans explication|chercher le lien manquant entre les evenements|faire lire le passage a une personne qui ne connait pas le plan|le recit devient plus facile a suivre
au centre d'archives|presenter un document|la fiche de presentation|la date exacte n'est pas connue|indiquer les elements qui permettent une estimation|distinguer la date certaine de la periode supposee|la presentation devient plus precise sur ses limites
au lieu de rencontre|preparer une activite collective|la disposition des tables|certains passages sont trop etroits|revoir la disposition en tenant compte des deplacements|demander aux participants si des obstacles restent presents|l'espace devient plus simple a utiliser
au bureau de redaction|preparer une annonce|le message|le lieu et l'heure sont disperses dans le texte|regrouper les informations pratiques|relire le message du point de vue d'un nouveau lecteur|l'annonce devient plus facile a utiliser
au domicile|organiser un rangement|les etageres|les objets les plus utiles sont difficiles a atteindre|classer les objets selon leur usage|essayer le rangement pendant quelques jours|le rangement peut etre ajuste a partir de l'experience
au club de jardinage|comparer des observations|les photographies des plantes|les images ont ete prises sous des angles differents|noter les conditions de prise de vue|utiliser un point de vue comparable pour la prochaine serie|les changements observes sont plus faciles a discuter
au bureau de coordination|repartir des taches|le tableau de suivi|une tache n'a pas de responsable identifie|demander qui peut la prendre en charge|faire confirmer le delai et la responsabilite|le suivi evite une attente sans interlocuteur
au petit salon du livre|preparer une rencontre|le programme|le titre d'une intervention est peu explicite|demander un titre qui indique le sujet|comparer le titre avec le contenu annonce|le public peut mieux choisir les rencontres
au club de randonnee|preparer une sortie|la description du parcours|les participants n'ont pas tous le meme rythme|discuter des attentes avant de choisir le parcours|prevoir des points de regroupement adaptes|le groupe partage un plan plus realiste
au centre de ressources|organiser un pret|le registre|la date de retour n'a pas ete notee|retrouver l'information avec la personne concernee|inscrire une date confirmee plutot qu'une estimation presentee comme certaine|le suivi du pret devient plus clair
au petit journal local|rediger un compte rendu|les notes de reunion|une proposition est confondue avec une decision|relire les notes en distinguant les statuts|faire confirmer les decisions par les participants|le compte rendu restitue mieux la reunion
au club de mathematiques|expliquer un probleme|le schema|le calcul utilise une donnee qui n'est pas fournie|revenir a l'enonce et aux unites|distinguer une hypothese supplementaire des donnees certaines|la solution explique clairement ses conditions
au centre d'apprentissage|preparer un exercice|la consigne|les participants ignorent le resultat attendu|donner un exemple de production attendue|verifier que l'exemple ne revele pas toute la reponse|l'exercice devient plus accessible
au local associatif|preparer un bilan|les retours des participants|quelques avis visibles masquent les autres|rassembler les retours avec une methode commune|signaler combien de personnes ont repondu|le bilan presente mieux les limites de ses observations
'''

NAMES = ['lea', 'paul', 'sara', 'noe', 'lina', 'adam', 'emma', 'hugo', 'ines', 'malo', 'nora', 'samir', 'alice', 'lucas', 'maya', 'yanis', 'clara', 'theo', 'lou', 'eli']
TIMES = ['le matin', 'dans la matinee', 'apres le repas', 'en debut de journee', 'dans l\'apres-midi', 'avant la fermeture', 'un lundi', 'un mercredi', 'un vendredi', 'pendant une journee de travail']
CHECKS = [
    'une solution proposee doit encore etre verifiee dans la situation reelle.',
    'un exemple utile ne remplace pas toutes les informations sur la situation.',
    'si une information manque, il vaut mieux la demander que l\'inventer.',
    'les personnes presentes peuvent avoir des contraintes qui n\'ont pas encore ete exprimees.',
    'la prochaine etape doit etre assez claire pour que chacun sache quoi faire.',
    'noter la raison du choix permettra de comprendre plus tard pourquoi il a ete fait.',
    'une premiere version peut etre ajustee apres avoir observe son utilisation.',
    'il est utile de distinguer un resultat constate et un resultat seulement attendu.',
]
LESSONS = [line.split('|') for line in LESSONS_TEXT.strip().splitlines()]
SCENARIOS = [line.split('|') for line in SCENARIOS_TEXT.strip().splitlines()]
assert all(len(x) == 5 for x in LESSONS)
assert all(len(x) == 7 for x in SCENARIOS)


def normalize(text):
    text = text.lower().replace('œ', 'oe').replace('æ', 'ae')
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
    text = text.translate(str.maketrans({'’': "'", '«': '"', '»': '"', '—': '-'}))
    return re.sub(r'[ \t]+', ' ', text).strip()


def scenario(rng, index):
    location, task, obj, problem, first, check, result = SCENARIOS[index]
    a, b = rng.sample(NAMES, 2)
    moment = rng.choice(TIMES)
    note = rng.choice(CHECKS)
    style = rng.randrange(6)
    if style == 0:
        return f'{moment}, {a} et {b} se retrouvent {location} pour {task}. le travail porte sur {obj}. une difficulte apparait : {problem}. {a} propose de {first}. {b} prefere commencer par comprendre ce qui est deja connu. apres en avoir discute, les deux personnes decident de {check}. cette verification change leur maniere de travailler. finalement, {result}. elles notent le choix retenu pour pouvoir l\'expliquer. la situation rappelle que {note}'
    if style == 1:
        return f'{location}, {a} discute avec {b}.\n"que devons-nous faire ?"\n"nous voulons {task}, mais {problem}."\n"sur quoi pouvons-nous nous appuyer ?"\n"nous avons {obj}. nous pouvons commencer par {first}."\n"comment saurons-nous si cela convient ?"\n"il faudra {check}."\nles deux personnes conviennent de cette prochaine etape. apres le travail, {result}. avant de partir, elles gardent une remarque dans leurs notes : {note}'
    if style == 2:
        return f'compte rendu de {a}. lieu : {location}. objectif : {task}. le document ou le materiel examine est {obj}. difficulte observee : {problem}. action proposee : {first}. controle a effectuer : {check}. resultat du travail : {result}. {b} rappelle que le resultat doit etre decrit sans exageration. une correction est apportee a la formulation du compte rendu. remarque finale : {note}'
    if style == 3:
        return f'question : comment {task} lorsque {problem} ?\nreponse : une premiere piste consiste a {first}. il faut ensuite {check}. dans cet exemple situe {location}, {a} et {b} utilisent {obj} pour clarifier leur travail. le resultat observe est le suivant : {result}. cette proposition concerne la situation decrite ; elle ne constitue pas une recette qui fonctionnerait dans tous les cas. pour l\'adapter, il faut connaitre les contraintes locales. en particulier, {note}'
    if style == 4:
        return f'{moment}, {a} arrive {location}. il est question de {task}. {b} explique la difficulte : {problem}. au lieu de choisir une solution immediatement, le groupe regarde {obj}. une possibilite est de {first}. une autre question doit etre resolue : comment {check} ? les participants prennent le temps de le faire. a la fin de leur travail, {result}. personne ne pretend que tous les problemes sont regles. le bilan conserve cette reserve : {note}'
    return f'notes pour un atelier {location}. le cas propose demande de {task}. les participants savent que {problem}. ils disposent de {obj}.\nconsigne : expliquer pourquoi il serait utile de {first}, puis indiquer comment {check}.\nexemple de conclusion : apres cette demarche, {result}.\n{a} et {b} comparent leurs propositions. les formulations different, mais chacune doit relier la difficulte aux actions choisies. une remarque sert a relire les reponses : {note}'


def lesson(rng, index):
    topic, explanation, example, confusion, question = LESSONS[index]
    style = rng.randrange(5)
    framing = rng.choice(CHECKS)
    if style == 0:
        return f'sujet : {topic}.\n{explanation}\nexemple : {example}\npoint de vigilance : {confusion}\nquestion pour poursuivre : {question}\npour preparer une reponse, il est utile de revenir a la definition et de distinguer les faits des hypotheses. {framing}'
    if style == 1:
        return f'question : que faut-il comprendre a propos du sujet "{topic}" ?\nreponse : {explanation}\nquestion : peut-on donner un exemple ?\nreponse : {example}\nquestion : quelle confusion faut-il eviter ?\nreponse : {confusion}\nquestion : {question}\nreponse : la question invite a examiner les conditions de l\'exemple ; une formule apprise seule ne suffit pas toujours. {framing}'
    if style == 2:
        a, b = rng.sample(NAMES, 2)
        return f'{a} prepare une courte presentation sur le sujet "{topic}". la premiere version contient cette explication : {explanation} {b} demande un exemple pour mieux suivre. {a} ajoute : {example} la discussion fait apparaitre une precision importante : {confusion} avant de terminer, les deux personnes inscrivent une question au tableau : {question} cette question servira a verifier que les lecteurs peuvent utiliser l\'explication. {framing}'
    if style == 3:
        return f'fiche de lecture : {topic}.\nidee principale : {explanation}\nillustration : {example}\nlimite d\'une interpretation trop rapide : {confusion}\nconsigne de reformulation : expliquer l\'idee principale avec ses propres mots, puis choisir un autre exemple sans changer le sens. terminer par cette question : {question}\nune bonne reformulation doit conserver les relations importantes, meme si les mots sont differents. {framing}'
    return f'pour aborder le sujet "{topic}", commencons par une idee simple : {explanation} l\'exemple suivant permet de concretiser l\'explication : {example} il faut toutefois rester attentif a une confusion : {confusion} un lecteur peut alors se demander : {question} la suite du travail consiste a preciser les conditions, a chercher des observations et a expliquer les liens entre elles. {framing}'


def exercise(rng):
    a = rng.choice(NAMES)
    kind = rng.randrange(12)
    if kind == 0:
        n, price = rng.randint(2, 30), rng.randint(2, 25)
        return f'{a} achete {n} cahiers a {price} euros chacun. quel est le prix total ?\nle prix unitaire est identique pour chaque cahier. il faut donc multiplier le nombre de cahiers par le prix d\'un cahier : {n} x {price} = {n * price}. le prix total est de {n * price} euros. pour controler le resultat, on peut verifier que l\'unite finale est une somme d\'argent. aucun autre achat ni reduction n\'est indique dans l\'enonce.'
    if kind == 1:
        total, used = rng.randint(40, 250), rng.randint(5, 35)
        return f'une boite contient {total} fiches. {a} en distribue {used}. combien de fiches restent dans la boite ?\nil faut soustraire la quantite distribuee de la quantite initiale : {total} - {used} = {total - used}. il reste {total - used} fiches. une verification consiste a ajouter les fiches restantes et les fiches distribuees : on doit retrouver {total}. ce calcul suppose qu\'aucune autre fiche n\'a ete ajoutee ou retiree.'
    if kind == 2:
        groups, size = rng.randint(2, 18), rng.randint(3, 16)
        return f'{a} repartit {groups * size} jetons de facon egale entre {groups} groupes. combien de jetons chaque groupe recoit-il ?\nla division {groups * size} / {groups} donne {size}. chaque groupe recoit donc {size} jetons. il n\'y a pas de reste dans cet exemple. on peut controler la repartition en multipliant {size} par {groups} : on retrouve {groups * size}. le nombre de groupes et le nombre de jetons par groupe ont des roles differents.'
    if kind == 3:
        base, factor, amount = rng.choice([2, 3, 4, 5, 6]), rng.randint(2, 5), rng.randrange(50, 251, 10)
        return f'une recette pour {base} personnes utilise {amount} grammes de farine. {a} veut preparer la meme recette pour {base * factor} personnes. quelle quantite de farine faut-il prevoir ?\nle nombre de personnes est multiplie par {factor}. en gardant les memes proportions, la farine est aussi multipliee par {factor} : {amount} x {factor} = {amount * factor}. il faut {amount * factor} grammes de farine. cette adaptation des quantites ne permet pas, a elle seule, de determiner un nouveau temps de cuisson.'
    if kind == 4:
        h, minute, duration = rng.randint(7, 18), rng.choice([0, 10, 15, 20, 30, 40]), rng.randint(20, 150)
        end = h * 60 + minute + duration
        return f'un atelier commence a {h:02d}:{minute:02d} et dure {duration} minutes. a quelle heure se termine-t-il ?\non ajoute {duration} minutes a l\'heure de debut. pour eviter une confusion, on peut convertir l\'heure de debut en minutes : {h} x 60 + {minute} = {h * 60 + minute}. apres addition, on obtient {end} minutes depuis minuit, soit {end // 60:02d}:{end % 60:02d}. l\'atelier se termine donc a {end // 60:02d}:{end % 60:02d}, en supposant qu\'il n\'y a pas de pause supplementaire.'
    if kind == 5:
        length, width = rng.randint(3, 24), rng.randint(2, 18)
        return f'un rectangle mesure {length} centimetres de longueur et {width} centimetres de largeur. calculer son perimetre et son aire.\nle perimetre additionne les longueurs des quatre cotes : 2 x ({length} + {width}) = {2 * (length + width)} centimetres. l\'aire multiplie la longueur par la largeur : {length} x {width} = {length * width} centimetres carres. les deux resultats n\'utilisent pas la meme unite, car une longueur et une surface ne sont pas la meme grandeur.'
    if kind == 6:
        count, percent = rng.randrange(20, 401, 20), rng.choice([5, 10, 15, 20, 25, 30, 40, 50])
        result = count * percent // 100
        return f'un atelier dispose de {count} feuilles. il utilise {percent} pour cent de ce stock. combien de feuilles sont utilisees ?\non calcule {count} x {percent} / 100 = {result}. l\'atelier utilise {result} feuilles et il en reste {count - result}. le pourcentage decrit ici une part du stock initial. on peut verifier que la quantite utilisee et la quantite restante donnent ensemble {count} feuilles.'
    if kind == 7:
        center, delta = rng.randint(8, 35), rng.randint(1, 7)
        return f'trois mesures donnent {center - delta}, {center} et {center + delta}. quelle est leur moyenne ?\non additionne les trois valeurs : {center - delta} + {center} + {center + delta} = {3 * center}. on divise ensuite cette somme par 3. la moyenne est {center}. dans cet exemple, les deux valeurs extremes sont placees a la meme distance de la valeur centrale. la moyenne ne montre pas a elle seule cet ecart entre les valeurs.'
    if kind == 8:
        meters = rng.randint(2, 120)
        return f'{a} mesure une longueur de {meters} metres. quelle est cette longueur en centimetres ?\nun metre correspond a 100 centimetres. il faut donc multiplier {meters} par 100 : on obtient {meters * 100} centimetres. la longueur physique ne change pas ; seul le nombre qui l\'exprime change avec l\'unite. pour revenir aux metres, on divise {meters * 100} par 100 et on retrouve {meters}.'
    if kind == 9:
        initial, more, removed = rng.randint(10, 90), rng.randint(2, 40), rng.randint(1, 9)
        return f'une etagere contient {initial} livres. on ajoute {more} livres, puis on en retire {removed}. combien de livres reste-t-il ?\napres l\'ajout, il y a {initial} + {more} = {initial + more} livres. apres le retrait, il reste {initial + more} - {removed} = {initial + more - removed} livres. les deux actions sont decrites dans l\'ordre pour faciliter la verification. on conserve la meme unite a chaque etape : un nombre de livres.'
    if kind == 10:
        x, y = rng.randint(2, 35), rng.randint(2, 35)
        return f'{a} prepare {x} cartes pour un premier groupe et {y} pour un second groupe. combien de cartes faut-il en tout ?\non rassemble les deux quantites : {x} + {y} = {x + y}. il faut {x + y} cartes. si les deux groupes doivent recevoir une carte par personne, ce resultat correspond au nombre total de personnes indique. il ne faut pas multiplier {x} par {y}, car les quantites decrivent ici deux groupes a reunir, pas un nombre de paquets identiques.'
    speed, hours = rng.randint(3, 8), rng.randint(2, 5)
    return f'dans un exercice simplifie, une personne avance a vitesse constante de {speed} kilometres par heure pendant {hours} heures. quelle distance parcourt-elle ?\nla distance est le produit de la vitesse et de la duree : {speed} x {hours} = {speed * hours} kilometres. ce resultat suppose une vitesse constante et aucune pause. pour un trajet reel, il faudrait examiner les arrets et les variations de vitesse. le calcul repond au modele decrit, pas a toutes les conditions possibles d\'une promenade.'


def build():
    rng = random.Random(SEED)
    docs, seen, counts = [], set(), Counter()
    # Equal coverage of authored topics; number exercises provide computed variants.
    scenario_order, lesson_order = [], []
    chars = 0
    while chars < TARGET:
        kind = rng.choices(['situation', 'explication', 'calcul'], weights=[45, 35, 20])[0]
        if kind == 'situation':
            if not scenario_order:
                scenario_order = list(range(len(SCENARIOS)))
                rng.shuffle(scenario_order)
            body = scenario(rng, scenario_order.pop())
        elif kind == 'explication':
            if not lesson_order:
                lesson_order = list(range(len(LESSONS)))
                rng.shuffle(lesson_order)
            body = lesson(rng, lesson_order.pop())
        else:
            body = exercise(rng)
        body = normalize(body)
        if body in seen:
            continue
        seen.add(body)
        docs.append(body)
        counts[kind] += 1
        chars += len(body) + 2
    text = '\n\n'.join(docs) + '\n'
    assert text.isascii() and text == text.lower()
    assert len(text) >= TARGET
    output = ROOT / 'data/processed/generaliste1m.txt'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Le corpus existe deja : {output}')
    output.write_text(text, encoding='utf-8')
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    freq = Counter(sentences)
    metadata = {
        'prepared_on': '2026-10-11', 'type': 'synthetic_controlled_templates',
        'source': 'Authored knowledge records and fictional situations in scripts/prepare_generaliste.py; arithmetic answers calculated in Python.',
        'seed': SEED, 'target_characters': TARGET,
        'characters': len(text), 'words_whitespace': len(text.split()),
        'vocabulary_size': len(set(text)), 'documents': len(docs),
        'document_counts_by_type': dict(counts),
        'authored_explanation_topics': len(LESSONS), 'authored_scenario_families': len(SCENARIOS),
        'exact_duplicate_documents': len(docs) - len(seen),
        'sentence_segments': len(sentences), 'unique_sentence_segments': len(freq),
        'repeated_sentence_occurrences': sum(n - 1 for n in freq.values()),
        'sha256': hashlib.sha256(text.encode()).hexdigest(),
        'normalization': 'lowercase ASCII, no accents, paragraph-separated documents',
        'limitations': [
            'Repeated sentence structures and reused authored facts; this is not one million independently authored characters.',
            'Near-duplicate template variants may occur across the automatic 90/10 train-validation split.',
            'Arithmetic examples teach text patterns; they do not guarantee arithmetic competence.',
            'Not instruction-tuned chat data; no current news, professional advice or comprehensive factual coverage.',
        ],
    }
    (ROOT / 'data/processed/generaliste1m_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
