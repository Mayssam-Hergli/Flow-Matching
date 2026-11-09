# -*- coding: utf-8 -*-
# =============================================================================
#  outils/verifier_physique.py : vérifications physiques (outil de développement, NON livré)
#
#  Utilise la fonction du professeur calcul_AF_performance_metrics, lue directement dans
#  prof/test_model_FFNN.py (le fichier original, pas une copie), et vérifie :
#    a. le cas test du cours : [6,6,6,5,5], theta0 = 112.36° -> G ~ 28.94 dB, SSL ~ -9.58 dB, HPBW ~ 76-77° ;
#    b. propriété P1 sur le jeu généré : G = 20*log10(N1+...+N5) pour chaque échantillon (tolérance 0.01 dB) ;
#    c. propriété P2 : pour 20 architectures aléatoires, theta0 et 180°-theta0 donnent les mêmes G, SSL, HPBW ;
#    d. cohérence des données : pas d'architecture nulle, N entiers dans 0..10, theta0 dans [0, 180],
#       SSL <= 0, aucun NaN ;
#    e. (bonus) le jeu généré par livrable/dataset_FM.py est identique à celui de prof/dataset.py
#       (seulement si prof/Minput.npy existe et a le même nombre d'échantillons).
#
#  Lancement (depuis la racine du dépôt) : .venv\Scripts\python.exe outils\verifier_physique.py
#  Code de sortie 0 si tout passe, 1 sinon.
# =============================================================================
import ast
import os
import sys
import numpy as np
from scipy.signal import find_peaks

racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dossier_livrable = os.path.join(racine, 'livrable')
dossier_prof = os.path.join(racine, 'prof')

# --- Paramètres physiques (CLAUDE.md §6) ---
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
max_elements = 10


# --- Chargement de la fonction du professeur depuis son fichier original ---
def charger_fonction_prof():
    chemin = os.path.join(dossier_prof, 'test_model_FFNN.py')
    source = open(chemin, encoding='utf-8').read()
    arbre = ast.parse(source)
    noeud = next(n for n in arbre.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'calcul_AF_performance_metrics')
    code_fonction = ast.get_source_segment(source, noeud)
    # Variables globales attendues par la fonction (mêmes valeurs que dans le script du professeur)
    espace = {
        'np': np, 'find_peaks': find_peaks,
        'r0': r0, 'delta_r': delta_r, 'max_rings': max_rings,
        'k': 2 * np.pi / lambda_, 'phi0': 0, 'phi': 0,
        'theta': np.linspace(0, 2 * np.pi, 1000),
        'theta0': None,
    }
    exec(code_fonction, espace)
    return espace


espace_prof = charger_fonction_prof()


def metriques(architecture, theta0deg):
    """Renvoie (G, SSL, HPBW) calculés par la fonction du professeur."""
    espace_prof['theta0'] = np.deg2rad(theta0deg)   # la fonction lit la globale theta0 (radians)
    _, HPBW, G, SSL = espace_prof['calcul_AF_performance_metrics'](np.array(architecture))
    return G, SSL, HPBW


resultats = []   # liste de (nom, réussi)


def bilan(nom, reussi):
    resultats.append((nom, reussi))
    print(f"  ==> {'PASS' if reussi else 'FAIL'}")


print("=" * 80)
print("VÉRIFICATIONS PHYSIQUES (fonction du professeur : prof/test_model_FFNN.py)")
print("=" * 80)

# -----------------------------------------------------------------------------
# a. Cas test du cours
# -----------------------------------------------------------------------------
print("\n[a] Cas test du cours : architecture [6,6,6,5,5], theta0 = 112.36°")
G, SSL, HPBW = metriques([6, 6, 6, 5, 5], 112.36)
print(f"  Valeurs obtenues : G = {G:.4f} dB   SSL = {SSL:.4f} dB   HPBW = {HPBW:.4f}°")
print("  Valeurs attendues : G ~ 28.94 dB   SSL ~ -9.58 dB   HPBW ~ 76-77°")
ok_G = abs(G - 28.94) <= 0.01
ok_SSL = abs(SSL - (-9.58)) <= 0.01
ok_HPBW = 76.0 <= HPBW <= 77.0
print(f"  G à 0.01 dB près : {ok_G}   SSL à 0.01 dB près : {ok_SSL}   HPBW dans [76, 77] : {ok_HPBW}")
bilan("a. cas test du cours", ok_G and ok_SSL and ok_HPBW)

# -----------------------------------------------------------------------------
# Chargement du jeu de données généré par livrable/dataset_FM.py
# -----------------------------------------------------------------------------
Minput = np.load(os.path.join(dossier_livrable, 'Minput.npy'))
Moutput = np.load(os.path.join(dossier_livrable, 'Moutput.npy'))
nb = Minput.shape[1]
print(f"\nJeu de données chargé : Minput {Minput.shape}, Moutput {Moutput.shape}")

# -----------------------------------------------------------------------------
# b. Propriété P1 : G = 20*log10(N_tot)
# -----------------------------------------------------------------------------
print("\n[b] Propriété P1 : G = 20*log10(N1+...+N5) pour chaque échantillon (tolérance 0.01 dB)")
N_tot = Moutput.sum(axis=0)
G_theorique = 20 * np.log10(N_tot)
ecart_P1 = np.abs(Minput[0] - G_theorique)
nb_hors_tol = int(np.sum(ecart_P1 > 0.01))
print(f"  Écart max = {ecart_P1.max():.6f} dB   écart moyen = {ecart_P1.mean():.6f} dB   "
      f"échantillons hors tolérance : {nb_hors_tol}/{nb}")
bilan("b. propriété P1 (G = 20 log10 N_tot)", nb_hors_tol == 0)

# -----------------------------------------------------------------------------
# c. Propriété P2 : symétrie theta0 <-> 180° - theta0
# -----------------------------------------------------------------------------
print("\n[c] Propriété P2 : theta0 et 180°-theta0 donnent les mêmes G, SSL, HPBW (20 architectures)")
rng = np.random.RandomState(46)
ecarts = []
for i in range(20):
    while True:
        arch = rng.randint(0, max_elements + 1, size=max_rings)
        if arch.sum() > 0:
            break
    t0 = rng.uniform(0, 180)
    m1 = metriques(arch, t0)
    m2 = metriques(arch, 180 - t0)
    ecart = np.abs(np.array(m1) - np.array(m2))
    ecarts.append(ecart)
    print(f"  {i+1:2d}. {arch}  theta0 = {t0:7.2f}° / {180 - t0:7.2f}°  ->  "
          f"G {m1[0]:6.2f}/{m2[0]:6.2f}  SSL {m1[1]:7.2f}/{m2[1]:7.2f}  HPBW {m1[2]:6.2f}/{m2[2]:6.2f}")
ecarts = np.array(ecarts)
print(f"  Écart max : G = {ecarts[:, 0].max():.2e} dB   SSL = {ecarts[:, 1].max():.2e} dB   "
      f"HPBW = {ecarts[:, 2].max():.2e}°")
bilan("c. propriété P2 (theta0 / 180-theta0)", bool(np.all(ecarts <= 1e-6)))

# -----------------------------------------------------------------------------
# d. Cohérence des données
# -----------------------------------------------------------------------------
print("\n[d] Cohérence des données")
controles = {
    "formes 4 x N et 5 x N": Minput.shape == (4, nb) and Moutput.shape == (5, nb),
    "aucun NaN / infini": bool(np.all(np.isfinite(Minput)) and np.all(np.isfinite(Moutput))),
    "aucune architecture entièrement nulle": bool(np.all(Moutput.sum(axis=0) > 0)),
    "tous les N sont entiers": bool(np.all(Moutput == np.round(Moutput))),
    "tous les N dans 0..10": bool(Moutput.min() >= 0 and Moutput.max() <= max_elements),
    "theta0 dans [0, 180]": bool(Minput[3].min() >= 0 and Minput[3].max() <= 180),
    "SSL <= 0": bool(np.all(Minput[1] <= 0)),
}
for nom, ok in controles.items():
    print(f"  {nom:40s} : {'oui' if ok else 'NON'}")
bilan("d. cohérence des données", all(controles.values()))

# -----------------------------------------------------------------------------
# e. (bonus) Identique au jeu du professeur
# -----------------------------------------------------------------------------
print("\n[e] (bonus) Jeu identique à celui de prof/dataset.py (même graine, même tirage)")
chemin_prof_in = os.path.join(dossier_prof, 'Minput.npy')
chemin_prof_out = os.path.join(dossier_prof, 'Moutput.npy')
if not (os.path.exists(chemin_prof_in) and os.path.exists(chemin_prof_out)):
    print("  prof/Minput.npy absent (lancer prof/dataset.py d'abord) : vérification ignorée")
else:
    Minput_prof = np.load(chemin_prof_in)
    Moutput_prof = np.load(chemin_prof_out)
    if Minput_prof.shape != Minput.shape:
        print(f"  Tailles différentes ({Minput_prof.shape} vs {Minput.shape}) : vérification ignorée")
    else:
        identique_in = bool(np.array_equal(Minput_prof, Minput))
        identique_out = bool(np.array_equal(Moutput_prof, Moutput))
        print(f"  Minput identique bit à bit : {identique_in}   "
              f"(écart max {np.abs(Minput_prof - Minput).max():.2e})")
        print(f"  Moutput identique bit à bit : {identique_out}")
        bilan("e. identique au jeu du professeur", identique_in and identique_out)

# -----------------------------------------------------------------------------
# Bilan
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("BILAN")
for nom, ok in resultats:
    print(f"  {'PASS' if ok else 'FAIL'}  {nom}")
tout_ok = all(ok for _, ok in resultats)
print(f"\n{'TOUTES LES VÉRIFICATIONS PASSENT' if tout_ok else 'AU MOINS UNE VÉRIFICATION ÉCHOUE'}")
print("=" * 80)
sys.exit(0 if tout_ok else 1)
