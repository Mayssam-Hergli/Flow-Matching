# Journal de bord : Flow Matching pour réseaux d'antennes circulaires

Ce fichier est tenu **en ajout seulement**. Chaque étape y consigne ce qui a été fait, les fichiers touchés,
les décisions prises (et pourquoi), les résultats réels et les questions ouvertes.

---

## Étape 0 : squelette du dépôt + baseline FFNN du professeur

*Date : 2026-11-09*

### Ce qui a été fait
1. Création du squelette décrit dans `CLAUDE.md` §3 : dossiers `livrable/` et `outils/` (avec `.gitkeep`),
   `requirements.txt`, `.gitignore`, `README.md` (en français), ce `JOURNAL.md`.
2. Création d'un environnement virtuel `.venv/` à la racine (ignoré par git) avec Python 3.12, et installation
   des dépendances (`pip install torch numpy scipy matplotlib`).
3. Exécution du pipeline du professeur **sans aucune modification**, dans `prof/`, en mode sans affichage
   (`$env:MPLBACKEND="Agg"`), dans l'ordre :
   `dataset.py` → `repartition_training_test.py` → `FFNN_training.py` → `test_model_FFNN.py`.
   Les 4 scripts se terminent avec le code de sortie 0. Les fichiers générés dans `prof/`
   (`*.npy`, `*.mat`, `ffnn_model.pth`, `normalization_params.npz`) sont ignorés par git.

### Fichiers créés
- `README.md`, `JOURNAL.md`, `requirements.txt`, `.gitignore`, `livrable/.gitkeep`, `outils/.gitkeep`.
- `CLAUDE.md`, `.claude/settings.json`, `prof/` et `rapport/` sont ajoutés au dépôt **tels quels**
  (premier commit du dépôt). Le statut de l'étape 0 dans `CLAUDE.md` §9 était déjà ☑.

### Environnement
| Composant | Version |
|---|---|
| Python | 3.12.10 (Windows 11, 64 bits) |
| torch | 2.15.0+cpu |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

Ces versions sont épinglées dans `requirements.txt`.

### Décisions
- **torch en version CPU** : c'est la roue installée par défaut par `pip install torch` sous Windows, donc celle
  que le professeur obtiendra avec `pip install -r requirements.txt`. La machine a une RTX 4060, mais le FFNN
  s'entraîne en ~10 s sur CPU. On reverra la question si l'entraînement du Flow Matching devient trop lent (étape 4).
- **`.gitignore`** : on ignore les caches Python, les environnements virtuels, tous les fichiers générés
  (`*.npy`, `*.mat`, `*.pth`, `*.npz`, `*.png`), les `.txt` de résultats dans `prof/` et `livrable/`, ainsi que
  `livrable/fm_config.json` (fichier généré par `FM_training.py`). Les fichiers entraînés finaux seront ajoutés
  avec `git add -f` à l'étape 5. On ignore aussi `.claude/settings.local.json` (réglages personnels) et les
  fichiers auxiliaires LaTeX dans `rapport/`.
- `outils/verifier_blocs_communs.py` n'existe pas encore : pas de vérification des blocs communs à cette étape.

### Résultats : baseline FFNN du professeur

**Dataset** (`dataset.py`) : `nb_samples = 1000`, graine `46`.
- `Minput` : shape `(4, 1000)` ; `Moutput` : shape `(5, 1000)`.
- Aucune architecture entièrement nulle. Nombre moyen d'éléments par anneau ≈ 5 (uniforme sur 0..10).
- Plages de `Minput` : G ∈ [12.04, 33.44] dB, SSL ∈ [−38.69, −1.35] dB, HPBW ∈ [11.17, 113.51]°, theta0 ∈ [0.06, 179.68]°.
- Répartition (`repartition_training_test.py`, 10 % test, graine 46) : training `(4, 900)` / `(5, 900)`,
  test `(4, 100)` / `(5, 100)`.

**Entraînement** (`FFNN_training.py`) : FFNN 4 → 14 → 20 → 8 → 5 (ReLU), normalisation min-max,
Adam lr = 0.005, batch 64, 100 époques.
- Loss époque 1 : 3.4232 ; époque 50 : 1.1505. **Loss finale (époque 100) : 1.1614.**
- **Durée d'entraînement : 9.97 s** (CPU).
- Attention : la « loss » affichée est la **somme** des MSE des 15 mini-batchs de l'époque
  (900 / 64 → 15 batchs), pas une moyenne. MSE moyenne par batch ≈ 1.16 / 15 ≈ 0.077 (en unités normalisées [0, 1]).

**Test** (`test_model_FFNN.py`) : sortie console complète (échantillon de test `indice_test = 10`) :

```
Architecture référence  : [1 8 6 5 2]
Architecture prédite    : [4 4 4 5 5]

theta0deg                 : 30.40°

--- Performances Référence ---
Gain du lobe principal : 26.85 dB
True Side Lobe Gain   : -7.44 dB
HPBW                  : 17.30°

--- Performances Prédiction ---
Gain du lobe principal : 26.85 dB
True Side Lobe Gain   : -6.77 dB
HPBW                  : 17.30°

=== Erreurs absolues (MAE) entre prédiction et référence ===
Erreur Gain Lobe Principal : 0.00 dB
Erreur True SSL Gain      : 0.66 dB
Erreur HPBW               : 0.00 degrés

=== Erreurs relatives de prédiction (%) ===
Erreur relative de prédiction Gain Lobe Principal : 0.00 %
Erreur relative de prédiction True SSL Gain      : 1.71 %
Erreur relative de prédiction HPBW               : 0.00 %

Erreur globale de prédiction pondérée : 0.57 %
```

### Observations
- **Le résultat du FFNN n'est pas reproductible.** `FFNN_training.py` fixe la graine de numpy mais **pas celle
  de torch** (initialisation des poids, mélange du DataLoader). Deux exécutions donnent deux modèles différents.
  Le dataset et la répartition, eux, sont reproductibles (graine numpy 46).
- Le script de test n'évalue qu'**un seul échantillon** (`indice_test = 10`), pas tout le jeu de test.
  L'erreur globale de 0.57 % repose donc sur un seul exemple et dépend de l'exécution.
- Le FFNN prédit une architecture proche de la moyenne du dataset ([4 4 4 5 5], moyenne ≈ [5 5 5 5 5]),
  très loin de la référence [1 8 6 5 2]. Pourtant, les performances sont presque identiques (même G, même HPBW,
  SSL à 0.66 dB près). Cela illustre la **non-unicité du problème inverse** : plusieurs architectures donnent
  des spécifications voisines. Un modèle de régression (MSE) converge vers une « moyenne » des solutions ;
  c'est la motivation du Flow Matching, qui génère plusieurs candidates.
- `FFNN_training.py` et `test_model_FFNN.py` n'enregistrent pas leurs figures (seulement `plt.show()`), donc aucun
  `.png` n'est produit en mode `Agg`. Chaque `plt.show()` émet un simple avertissement
  `FigureCanvasAgg is non-interactive`, sans conséquence.

### Questions ouvertes
- Pour comparer FM et FFNN à l'étape 3, il faudra une métrique moyennée sur **tout** le jeu de test
  (et idéalement sur plusieurs entraînements du FFNN), sinon la comparaison sera dominée par le bruit.
  Cela pourra se faire sans modifier `prof/` (par exemple un script dans `outils/`).
- Le rapport s'appelle `rapport/rapport_flow_matching (4).pdf`, alors que `CLAUDE.md` mentionne
  `rapport/rapport_flow_matching.pdf`. Faut-il renommer le fichier ?
