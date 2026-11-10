# -*- coding: utf-8 -*-
"""
outils/verifier_entrainement.py : contrôle rapide de l'entraînement (outil de développement, NON livré).

Ce n'est PAS l'évaluation de l'étape 3 (pas encore de métrique du professeur) : c'est un bilan de santé.
  a. reconstruit le réseau à partir de livrable/fm_config.json (la classe VelocityNet est lue dans le
     BLOC COMMUN de livrable/FM_training.py) et charge livrable/fm_model.pth            -> PASS / FAIL
  b. compare la perte de validation de l'époque 1 à celle de la meilleure époque
     (PASS si elle a baissé d'au moins 30 %)                                             -> PASS / FAIL
  c. pour 20 échantillons de TEST, génère 64 candidats chacun (Euler, 20 pas), puis revient aux entiers
     (échelle inverse, arrondi, limitation 0..10) ;
  d. calcule le total attendu N_tot = round(10^(G/20)) à partir de G (propriété P1) ;
  e. affiche : % de candidats dont le total vaut N_tot, % d'échantillons avec au moins un candidat correct,
     nombre moyen de candidats distincts, et 3 exemples détaillés.

Lancement (depuis la racine du dépôt) : .venv\\Scripts\\python.exe outils\\verifier_entrainement.py
Code de sortie 0 si a et b passent, 1 sinon.
"""
import json
import math
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn

racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dossier = os.path.join(racine, 'livrable')

NB_ECHANTILLONS = 20     # échantillons de test examinés
K = 64                   # candidats par échantillon
NB_PAS = 20              # pas d'Euler
GRAINE = 46

torch.manual_seed(GRAINE)
np.random.seed(GRAINE)
resultats = []


def bilan(nom, ok):
    resultats.append((nom, ok))
    print(f"  ==> {'PASS' if ok else 'FAIL'}")


print("=" * 80)
print("VÉRIFICATION DE L'ENTRAÎNEMENT (bilan de santé, pas l'évaluation de l'étape 3)")
print("=" * 80)

# -----------------------------------------------------------------------------
# a. Reconstruction du réseau depuis fm_config.json
# -----------------------------------------------------------------------------
print("\n[a] Reconstruction du réseau depuis fm_config.json et chargement de fm_model.pth")
source = open(os.path.join(dossier, 'FM_training.py'), encoding='utf-8').read()
m = re.search(r'^# ===== BLOC COMMUN : VelocityNet .*?$(.*?)^# ===== FIN BLOC COMMUN =====$', source,
              re.S | re.M)
espace = {'torch': torch, 'nn': nn, 'math': math}
exec(m.group(1), espace)
VelocityNet = espace['VelocityNet']

config = json.load(open(os.path.join(dossier, 'fm_config.json'), encoding='utf-8'))
model = VelocityNet(**config['modele'])
try:
    model.load_state_dict(torch.load(os.path.join(dossier, 'fm_model.pth'), map_location='cpu', weights_only=True))
    charge = True
except Exception as e:
    print(f"  Erreur de chargement : {e}")
    charge = False
model.eval()
nb_parametres = sum(p.numel() for p in model.parameters())
print(f"  Réglages du réseau lus dans fm_config.json : {config['modele']}")
print(f"  Paramètres : {nb_parametres} (fm_config.json annonce {config['resultats']['nb_parametres']})")
print(f"  Chargement strict des poids : {'réussi' if charge else 'ÉCHEC'}")
bilan("a. réseau reconstruit depuis fm_config.json", charge and nb_parametres == config['resultats']['nb_parametres'])

# -----------------------------------------------------------------------------
# b. La perte de validation a-t-elle baissé ?
# -----------------------------------------------------------------------------
print("\n[b] Perte de validation : époque 1 vs meilleure époque")
r = config['resultats']
val1, val_best, ep_best = r['loss_val'][0], r['meilleure_loss_val'], r['meilleure_epoque']
baisse = 100 * (val1 - val_best) / val1
print(f"  Époque 1 : {val1:.4f}   meilleure (époque {ep_best}) : {val_best:.4f}   baisse : {baisse:.1f} %")
print(f"  (pour mémoire, loss train : époque 1 = {r['loss_train'][0]:.4f}, dernière = {r['loss_train'][-1]:.4f})")
bilan("b. baisse de la perte de validation >= 30 %", baisse >= 30)

# -----------------------------------------------------------------------------
# c. Génération de candidats pour des échantillons de test
# -----------------------------------------------------------------------------
print(f"\n[c] Génération : {NB_ECHANTILLONS} échantillons de test x {K} candidats, Euler {NB_PAS} pas")
Minput_test = np.load(os.path.join(dossier, 'Minput_test.npy')).T[:NB_ECHANTILLONS]    # (20, 4)
Moutput_test = np.load(os.path.join(dossier, 'Moutput_test.npy')).T[:NB_ECHANTILLONS]  # (20, 5)
norm = np.load(os.path.join(dossier, 'fm_normalization_params.npz'))
cond_mean, cond_std, n_max = norm['cond_mean'], norm['cond_std'], int(norm['n_max'])

condition = np.column_stack([Minput_test[:, 0], Minput_test[:, 1], Minput_test[:, 2],
                             np.sin(np.deg2rad(Minput_test[:, 3]))])
condition = (condition - cond_mean) / cond_std
c = torch.tensor(np.repeat(condition, K, axis=0), dtype=torch.float32)          # (20*64, 4)

x = torch.randn(NB_ECHANTILLONS * K, config['modele']['dim_x'])                  # bruits de départ
h = 1.0 / NB_PAS
with torch.no_grad():
    for s in range(NB_PAS):                                                       # algorithme 2 (Euler)
        t = torch.full((x.shape[0],), s * h)
        x = x + h * model(x, t, c)
N = np.clip(np.round((x.numpy() + 1) * n_max / 2), 0, n_max).astype(int)          # retour aux entiers
candidats = N.reshape(NB_ECHANTILLONS, K, -1)
print(f"  Valeurs continues générées : min = {x.min().item():.3f}   max = {x.max().item():.3f}  "
      f"(échelle [-1, 1] attendue)")

# -----------------------------------------------------------------------------
# d. Total attendu d'après G (propriété P1)
# -----------------------------------------------------------------------------
N_tot_attendu = np.round(10 ** (Minput_test[:, 0] / 20)).astype(int)
print("\n[d] Total attendu N_tot = round(10^(G/20)) (propriété P1)")
print(f"  Contrôle : N_tot attendu = somme de l'architecture de référence pour "
      f"{int(np.sum(N_tot_attendu == Moutput_test.sum(axis=1)))}/{NB_ECHANTILLONS} échantillons")

# -----------------------------------------------------------------------------
# e. Bilan des candidats
# -----------------------------------------------------------------------------
print("\n[e] Bilan des candidats")
totaux = candidats.sum(axis=2)                                    # (20, 64)
bon_total = totaux == N_tot_attendu[:, None]
pct_candidats = 100 * bon_total.mean()
pct_echantillons = 100 * bon_total.any(axis=1).mean()
nb_distincts = [len({tuple(a) for a in candidats[i]}) for i in range(NB_ECHANTILLONS)]
nb_nuls = int((totaux == 0).sum())
ecart_total = np.abs(totaux - N_tot_attendu[:, None])

# Niveau du hasard : si le modèle ignorait la condition et tirait une architecture de la base au hasard,
# quelle proportion aurait le bon total ?
Moutput_training = np.load(os.path.join(dossier, 'Moutput_training.npy')).T
totaux_base = Moutput_training.sum(axis=1)
hasard = 100 * np.mean([np.mean(totaux_base == n) for n in N_tot_attendu])

print(f"  Candidats dont le total vaut N_tot              : {pct_candidats:5.1f} %  "
      f"({int(bon_total.sum())}/{bon_total.size})")
print(f"  Échantillons avec au moins un candidat correct  : {pct_echantillons:5.1f} %  "
      f"({int(bon_total.any(axis=1).sum())}/{NB_ECHANTILLONS})")
print(f"  Candidats distincts par échantillon (moyenne)   : {np.mean(nb_distincts):5.1f} sur {K}  "
      f"(min {min(nb_distincts)}, max {max(nb_distincts)})")
print(f"  Écart moyen |total - N_tot|                     : {ecart_total.mean():5.2f} éléments  "
      f"(médiane {np.median(ecart_total):.0f})")
print(f"  Candidats sans aucun élément                    : {nb_nuls}")
print(f"  Pour comparaison, niveau du hasard (architectures de la base tirées au hasard) : {hasard:.1f} %")

print("\n  Trois exemples (5 premiers candidats de chacun) :")
for i in range(3):
    G, SSL, HPBW, th0 = Minput_test[i]
    print(f"\n  Échantillon de test {i} : G = {G:.2f} dB, SSL = {SSL:.2f} dB, HPBW = {HPBW:.2f}°, "
          f"theta0 = {th0:.2f}°  ->  N_tot attendu = {N_tot_attendu[i]}")
    print(f"    Référence  : {Moutput_test[i].astype(int)}  (total {int(Moutput_test[i].sum())})")
    for j in range(5):
        a = candidats[i, j]
        print(f"    Candidat {j + 1} : {a}  (total {a.sum():2d}{', correct' if a.sum() == N_tot_attendu[i] else ''})")
    print(f"    Sur les {K} candidats : {int(bon_total[i].sum())} ont le bon total, {nb_distincts[i]} distincts")

# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BILAN")
for nom, ok in resultats:
    print(f"  {'PASS' if ok else 'FAIL'}  {nom}")
print("  (c à e : indicateurs affichés, sans seuil de réussite)")
print("=" * 80)
sys.exit(0 if all(ok for _, ok in resultats) else 1)
