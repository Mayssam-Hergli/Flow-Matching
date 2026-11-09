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

---

## Maintenance : nom du rapport + graines torch

*Date : 2026-11-09*

- `rapport/rapport_flow_matching (4).pdf` renommé en `rapport/rapport_flow_matching.pdf` (`git mv`), pour correspondre
  à `CLAUDE.md`. Il n'y a pas de fichier `.tex` dans le dépôt. Cela répond à la question ouverte de l'étape 0.
- `CLAUDE.md` §4, puce « Seeds » (modification autorisée explicitement) : tous nos scripts fixent la graine 46 pour
  **numpy et torch** (`np.random.seed(46)`, `torch.manual_seed(46)`). Le FFNN du professeur ne fixe pas la graine
  torch (constaté à l'étape 0) : pour les comparaisons, il sera entraîné plusieurs fois et résumé en moyenne ± écart-type.

---

## Étape 1 : dataset + répartition + vérifications physiques

*Date : 2026-11-09*

### Ce que fait chaque fichier
- **`livrable/dataset_FM.py`** (script livré 1/4) : tire `nb_samples` architectures aléatoires `[N1..N5]`
  (0..10, au moins un anneau actif) et des theta0 aléatoires dans [0, 180]°, calcule `[G, SSL, HPBW]` avec la
  fonction du professeur, et enregistre `Minput` (4 × N) et `Moutput` (5 × N) en `.npy` et `.mat` dans `livrable/`.
  Affiche l'avancement tous les 10 % et un résumé (formes, min/max de chaque ligne de `Minput`,
  nombre d'échantillons par nombre total d'éléments `N_tot`).
  `nb_samples = 1000` est en tête de script (commentaire : passera à 50 000 à l'étape 4).
- **`livrable/repartition_training_test_FM.py`** (script livré 2/4) : même comportement que
  `prof/repartition_training_test.py` (graine 46, 10 % test, mêmes noms de fichiers, `.npy` et `.mat`,
  clé `.mat` = nom de la variable).
- **`outils/verifier_physique.py`** (développement, non livré) : vérifie la physique avec la fonction du professeur
  **lue directement dans `prof/test_model_FFNN.py`** (pas une copie) : cas test du cours, propriétés P1 et P2,
  cohérence des données, et en bonus l'identité avec le jeu du professeur.
- **`outils/verifier_blocs_communs.py`** (développement, non livré) : trouve les BLOCS COMMUNS dans `livrable/*.py`,
  les regroupe par nom, vérifie que les copies sont identiques (en-tête compris, fins de ligne ignorées), signale un
  script annoncé dans l'en-tête qui existe mais ne contient pas le bloc, et vérifie que la fonction
  `calcul_AF_performance_metrics` copiée est identique à l'originale du professeur. Code de sortie 1 en cas d'échec.
  Testé « à l'envers » sur une copie du dépôt (dans un dossier temporaire) : en changeant `max_elements` dans un
  seul script et `threshold_dB` dans la fonction copiée, il détecte les deux écarts (diff affiché, code de sortie 1).

### Blocs communs créés
- `parametres_physiques` : `carrierFreq, c, lambda_, r0, delta_r, max_rings, max_elements` (CLAUDE.md §6).
  Présent dans `dataset_FM.py` et `repartition_training_test_FM.py` (§6 le veut identique dans chaque script,
  même si la répartition ne s'en sert pas) ; annoncé aussi pour `FM_training.py` et `test_model_FM.py`.
- `calcul_AF_performance_metrics` : les constantes dont la fonction a besoin (`k, phi0, phi, theta, theta_deg`,
  mêmes valeurs que chez le professeur) + la fonction **copiée mot pour mot** depuis `prof/test_model_FFNN.py`
  (insérée par programme, pas retapée). Annoncé pour `dataset_FM.py` et `test_model_FM.py`.
  Attention : la fonction du professeur lit la **variable globale `theta0`** (en radians) ; il faut l'affecter
  avant chaque appel. On ne la modifie pas (règle 2), on le documente dans le commentaire du bloc.

### Différences entre `dataset_FM.py` et `prof/dataset.py` (toutes sans effet sur les données)
1. **Calcul des métriques** : le professeur écrit le calcul en ligne dans la boucle de `dataset.py` ; nous appelons
   sa fonction `calcul_AF_performance_metrics` de `test_model_FFNN.py`. Les deux codes sont les mêmes ligne à ligne
   (seule nuance : la fonction boucle sur `len(elements_per_ring)` au lieu de `max_rings`, soit 5 dans les deux cas).
   Pourquoi : CLAUDE.md impose la fonction du professeur, et la même copie servira dans `test_model_FM.py`,
   donc le jeu de données et la vérification utilisent exactement le même code.
2. Les constantes (`k`, grille `theta`, `phi`, `phi0`) sont définies une fois avant la boucle (le professeur recalcule
   `theta` et `radii` à chaque itération ; mêmes valeurs).
3. Paramètres physiques encadrés en BLOC COMMUN ; commentaires et en-tête explicatifs en français.
4. Imports inutilisés retirés (`matplotlib`, `platform`, `sys`) ; ajout de `torch.manual_seed(46)` (règle CLAUDE.md §4,
   bien que torch ne soit pas utilisé dans ce script) et de `time` pour la durée.
5. Ajouts d'affichage : avancement tous les 10 %, durée, résumé final.
6. Caractères `→` et `✅` retirés des `print` : sous Windows, quand la sortie est redirigée vers un fichier, Python
   utilise l'encodage cp1252 qui ne contient pas ces caractères (risque de `UnicodeEncodeError`). Nos scripts ont été
   lancés sans `PYTHONIOENCODING` et fonctionnent.
7. Même ordre de tirage aléatoire (architecture puis theta0), même graine 46.
   **Résultat : avec 1000 échantillons, les 6 matrices produites (`Minput`, `Moutput` et les 4 matrices training/test)
   sont identiques bit à bit à celles du professeur** (contrôle [e] ci-dessous, et comparaison des 6 fichiers `.npy`).

### Formes des matrices générées (dans `livrable/`)
| Matrice | Forme | `.npy` | `.mat` (clé = nom) |
|---|---|---|---|
| `Minput` | (4, 1000) | oui | oui |
| `Moutput` | (5, 1000) | oui | oui |
| `Minput_training` | (4, 900) | oui | oui |
| `Moutput_training` | (5, 900) | oui | oui |
| `Minput_test` | (4, 100) | oui | oui |
| `Moutput_test` | (5, 100) | oui | oui |

Les `.mat` relus avec `scipy.io.loadmat` contiennent la bonne clé et les mêmes valeurs que les `.npy`.
Durées : `dataset_FM.py` 1.7 s de génération (≈ 10 s avec le chargement de torch), répartition ≈ 4 s.

### Sortie de `dataset_FM.py` (résumé)
```
=== Génération terminée en 1.71 secondes ===
Plage de chaque ligne de Minput :
  [0] G (gain du lobe principal, dB)   : min =   12.0411   max =   33.4419
  [1] SSL (lobe secondaire, dB)        : min =  -38.6890   max =   -1.3464
  [2] HPBW (degrés)                    : min =   11.1712   max =  113.5135
  [3] theta0 (degrés)                  : min =    0.0646   max =  179.6784

Nombre d'échantillons par nombre total d'éléments N_tot (min = 4, max = 47) :
  N_tot= 4 :     2  N_tot= 5 :     3  N_tot= 6 :     3  N_tot= 7 :     3  N_tot= 8 :     2
  N_tot= 9 :     6  N_tot=10 :     6  N_tot=11 :     9  N_tot=12 :     8  N_tot=13 :     9
  N_tot=14 :    20  N_tot=15 :    16  N_tot=16 :    25  N_tot=17 :    32  N_tot=18 :    32
  N_tot=19 :    38  N_tot=20 :    37  N_tot=21 :    50  N_tot=22 :    54  N_tot=23 :    59
  N_tot=24 :    52  N_tot=25 :    62  N_tot=26 :    54  N_tot=27 :    65  N_tot=28 :    48
  N_tot=29 :    46  N_tot=30 :    51  N_tot=31 :    41  N_tot=32 :    26  N_tot=33 :    27
  N_tot=34 :    27  N_tot=35 :    16  N_tot=36 :    14  N_tot=37 :    12  N_tot=38 :    14
  N_tot=39 :     8  N_tot=40 :     7  N_tot=41 :     5  N_tot=42 :     7  N_tot=43 :     2
  N_tot=44 :     1  N_tot=47 :     1
```

### Sortie complète de `outils/verifier_physique.py`
```
================================================================================
VÉRIFICATIONS PHYSIQUES (fonction du professeur : prof/test_model_FFNN.py)
================================================================================

[a] Cas test du cours : architecture [6,6,6,5,5], theta0 = 112.36°
  Valeurs obtenues : G = 28.9431 dB   SSL = -9.5801 dB   HPBW = 76.7568°
  Valeurs attendues : G ~ 28.94 dB   SSL ~ -9.58 dB   HPBW ~ 76-77°
  G à 0.01 dB près : True   SSL à 0.01 dB près : True   HPBW dans [76, 77] : True
  ==> PASS

Jeu de données chargé : Minput (4, 1000), Moutput (5, 1000)

[b] Propriété P1 : G = 20*log10(N1+...+N5) pour chaque échantillon (tolérance 0.01 dB)
  Écart max = 0.000710 dB   écart moyen = 0.000069 dB   échantillons hors tolérance : 0/1000
  ==> PASS

[c] Propriété P2 : theta0 et 180°-theta0 donnent les mêmes G, SSL, HPBW (20 architectures)
   1. [ 5  8  4 10  3]  theta0 =  136.45° /   43.55°  ->  G  29.54/ 29.54  SSL   -7.04/  -7.04  HPBW  23.06/ 23.06
   2. [0 2 8 0 7]  theta0 =   81.90° /   98.10°  ->  G  24.61/ 24.61  SSL   -5.79/  -5.79  HPBW  56.94/ 56.94
   3. [2 2 2 6 7]  theta0 =  156.07° /   23.93°  ->  G  25.57/ 25.57  SSL  -10.71/ -10.71  HPBW  13.33/ 13.33
   4. [6 0 2 4 8]  theta0 =  113.00° /   67.00°  ->  G  26.02/ 26.02  SSL   -3.36/  -3.36  HPBW  72.07/ 72.07
   5. [1 1 8 0 7]  theta0 =   47.23° /  132.77°  ->  G  24.61/ 24.61  SSL   -6.21/  -6.21  HPBW  19.10/ 19.10
   6. [0 1 5 0 0]  theta0 =   39.80° /  140.20°  ->  G  15.56/ 15.56  SSL   -3.72/  -3.72  HPBW  24.14/ 24.14
   7. [ 8  0 10  6  6]  theta0 =  112.04° /   67.96°  ->  G  29.54/ 29.54  SSL   -7.10/  -7.10  HPBW  74.59/ 74.59
   8. [ 0  2  7 10  9]  theta0 =   74.42° /  105.58°  ->  G  28.94/ 28.94  SSL   -8.13/  -8.13  HPBW  61.98/ 61.98
   9. [10  3  4  2  6]  theta0 =  138.59° /   41.41°  ->  G  27.96/ 27.96  SSL   -5.29/  -5.29  HPBW  21.62/ 21.62
  10. [6 3 4 6 3]  theta0 =   35.12° /  144.88°  ->  G  26.85/ 26.85  SSL   -7.68/  -7.68  HPBW  20.18/ 20.18
  11. [2 5 9 0 5]  theta0 =    3.27° /  176.73°  ->  G  26.44/ 26.44  SSL  -12.60/ -12.60  HPBW  15.86/ 15.86
  12. [5 2 3 8 4]  theta0 =   17.09° /  162.91°  ->  G  26.85/ 26.85  SSL   -8.08/  -8.08  HPBW  15.50/ 15.50
  13. [2 3 3 5 5]  theta0 =  163.53° /   16.47°  ->  G  25.11/ 25.11  SSL   -7.26/  -7.26  HPBW  14.05/ 14.05
  14. [9 3 9 6 6]  theta0 =   44.08° /  135.92°  ->  G  30.37/ 30.37  SSL   -8.25/  -8.25  HPBW  22.70/ 22.70
  15. [6 5 9 1 7]  theta0 =   73.32° /  106.68°  ->  G  28.94/ 28.94  SSL   -5.91/  -5.91  HPBW  69.19/ 69.19
  16. [5 4 5 7 9]  theta0 =   41.89° /  138.11°  ->  G  29.54/ 29.54  SSL   -9.98/  -9.98  HPBW  18.74/ 18.74
  17. [1 4 0 2 5]  theta0 =   56.00° /  124.00°  ->  G  21.58/ 21.58  SSL   -4.20/  -4.20  HPBW  22.70/ 22.70
  18. [6 2 1 1 7]  theta0 =   46.30° /  133.70°  ->  G  24.61/ 24.61  SSL   -6.56/  -6.56  HPBW  19.82/ 19.82
  19. [1 1 3 5 2]  theta0 =   24.70° /  155.30°  ->  G  21.58/ 21.58  SSL   -5.86/  -5.86  HPBW  13.33/ 13.33
  20. [0 2 0 5 4]  theta0 =   41.64° /  138.36°  ->  G  20.83/ 20.83  SSL   -4.38/  -4.38  HPBW  16.22/ 16.22
  Écart max : G = 0.00e+00 dB   SSL = 5.33e-15 dB   HPBW = 0.00e+00°
  ==> PASS

[d] Cohérence des données
  formes 4 x N et 5 x N                    : oui
  aucun NaN / infini                       : oui
  aucune architecture entièrement nulle    : oui
  tous les N sont entiers                  : oui
  tous les N dans 0..10                    : oui
  theta0 dans [0, 180]                     : oui
  SSL <= 0                                 : oui
  ==> PASS

[e] (bonus) Jeu identique à celui de prof/dataset.py (même graine, même tirage)
  Minput identique bit à bit : True   (écart max 0.00e+00)
  Moutput identique bit à bit : True
  ==> PASS

================================================================================
BILAN
  PASS  a. cas test du cours
  PASS  b. propriété P1 (G = 20 log10 N_tot)
  PASS  c. propriété P2 (theta0 / 180-theta0)
  PASS  d. cohérence des données
  PASS  e. identique au jeu du professeur

TOUTES LES VÉRIFICATIONS PASSENT
================================================================================
```

### Sortie complète de `outils/verifier_blocs_communs.py`
```
================================================================================
VÉRIFICATION DES BLOCS COMMUNS (livrable/)
================================================================================
  trouvé : bloc 'parametres_physiques' dans dataset_FM.py (lignes 45-54)
  trouvé : bloc 'calcul_AF_performance_metrics' dans dataset_FM.py (lignes 57-142)
  trouvé : bloc 'parametres_physiques' dans repartition_training_test_FM.py (lignes 38-47)

[PASS] bloc 'calcul_AF_performance_metrics' : 1 copie(s) dans dataset_FM.py
        une seule copie pour l'instant : rien à comparer
        scripts annoncés pas encore créés : test_model_FM.py

[PASS] bloc 'parametres_physiques' : 2 copie(s) dans dataset_FM.py, repartition_training_test_FM.py
        toutes les copies sont identiques
        scripts annoncés pas encore créés : FM_training.py, test_model_FM.py

Comparaison avec la fonction originale du professeur (prof/test_model_FFNN.py) :
  [PASS] dataset_FM.py : fonction identique à l'originale

================================================================================
TOUS LES BLOCS COMMUNS SONT COHÉRENTS
================================================================================
```

### Observations (ce qui surprend ou servira plus tard)
- **Le cas test du cours est retrouvé** au centième : G = 28.9431 dB (= 20·log10 28), SSL = −9.5801 dB, HPBW = 76.76°.
- **P1 : le gain G ne dépend que de N_tot** (écart max 0.0007 dB, dû à la grille de 1000 angles qui ne tombe pas
  exactement sur theta0). Conséquences : `G` ne prend qu'une quarantaine de valeurs distinctes, et donner `G` revient
  à imposer `N1+…+N5`. Le modèle génératif devra respecter cette somme ; c'est aussi un contrôle simple des candidates.
- **P2 : la symétrie theta0 ↔ 180°−theta0 est exacte** (écarts nuls ou ~1e−15). Cela confirme le choix de la
  condition `sin(theta0)` (CLAUDE.md §5) : deux theta0 symétriques sont physiquement le même problème.
- **N_tot est très déséquilibré** (somme de 5 uniformes, forme en cloche centrée sur 25) : seulement 13 échantillons
  avec N_tot ≤ 8 et 23 avec N_tot ≥ 40 sur 1000. Les petites et grandes architectures seront rares à l'entraînement ;
  c'est ce que vise l'amélioration « N_tot balancing » prévue à l'étape 4.
- Les 5 premières architectures du test P2 sont les mêmes que les 5 premiers échantillons du jeu (même graine,
  même ordre de tirage). Sans conséquence : P2 compare theta0 à 180°−theta0, ce qui n'est jamais dans le jeu.
- Vitesse : 1000 échantillons en 1.7 s, donc environ 1 min 30 pour 50 000 échantillons à l'étape 4.

### Questions ouvertes
- `torch.manual_seed(46)` impose d'importer torch dans `dataset_FM.py` et `repartition_training_test_FM.py`, qui ne
  s'en servent pas (~5 s de chargement en plus). On l'a fait pour respecter la règle « every script we write » ;
  à retirer si vous préférez limiter la règle aux scripts qui utilisent torch.
