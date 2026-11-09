# Flow Matching pour la conception de réseaux d'antennes circulaires

## Résumé
Ce projet traite la **conception inverse** d'un réseau d'antennes circulaires concentriques à 5 anneaux.
À partir de spécifications de rayonnement `[G, SSL, HPBW, theta0]` (gain du lobe principal, niveau du lobe secondaire,
ouverture à mi-puissance, direction de pointage), on cherche le nombre d'éléments par anneau `[N1..N5]`.
Le problème n'a pas de solution unique : plusieurs architectures donnent des diagrammes voisins.
Un modèle génératif de **Flow Matching conditionnel** propose donc plusieurs architectures candidates ;
chacune est vérifiée avec le code physique du professeur, et la meilleure est retenue.

## Équipe
- Eya Hammami
- Isra Abessy
- Mayssam Hergli
- Malek Soltany
- Senda Derouiche
- Adem Gharbi
- Mowafak Jendli

**Encadrant :** Pr. Naoufel Ismail — INSAT, année universitaire 2026/2027.

## Organisation du dépôt
```
├── CLAUDE.md        contrat d'architecture du projet (règles, conventions, étapes)
├── JOURNAL.md       journal de bord : ce qui a été fait à chaque étape, décisions, résultats
├── README.md        ce fichier
├── requirements.txt dépendances Python (versions testées)
├── prof/            les scripts originaux du professeur (jamais modifiés)
├── livrable/        les 4 scripts livrés (jeu de données, répartition, entraînement, test)
├── outils/          scripts d'aide au développement (non livrés)
└── rapport/         rapport du projet
```

## Installation
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Comment lancer
Les 4 scripts se lancent dans l'ordre, depuis le dossier `livrable/` :

```powershell
cd livrable
python dataset_FM.py                     # 1. génération du jeu de données      (à compléter à l'étape 5)
python repartition_training_test_FM.py   # 2. répartition entraînement / test   (à compléter à l'étape 5)
python FM_training.py                    # 3. entraînement du modèle            (à compléter à l'étape 5)
python test_model_FM.py                  # 4. test et sélection de l'architecture (à compléter à l'étape 5)
```

*Section à compléter à l'étape 5 (fichiers produits, durées, résultats attendus).*
