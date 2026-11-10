# -*- coding: utf-8 -*-
# =============================================================================
#  dataset_FM.py : génération du jeu de données (script 1/4 du projet Flow Matching)
#
#  Pour chaque échantillon :
#    1. on tire une architecture aléatoire [N1..N5] (nombre d'éléments par anneau, 0..10,
#       au moins un anneau actif) ;
#    2. on tire une direction de pointage theta0 aléatoire dans [0, 180]° ;
#    3. on calcule le diagramme de rayonnement et ses performances [G, SSL, HPBW]
#       avec la fonction du professeur calcul_AF_performance_metrics (copiée telle quelle).
#
#  Sorties (dans le dossier du script, formats .npy et .mat) :
#    Minput  (4 x nb_samples) : [G (dB) ; SSL (dB) ; HPBW (°) ; theta0 (°)]
#    Moutput (5 x nb_samples) : [N1 ; N2 ; N3 ; N4 ; N5]
#
#  Même physique, même tirage aléatoire et même graine que prof/dataset.py :
#  avec nb_samples = 1000, les matrices produites sont identiques à celles du professeur.
# =============================================================================
import numpy as np
from scipy.signal import find_peaks
import scipy.io
import os
import time


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()



# === Paramètres modifiables ===
nb_samples = 1000        # nombre d'échantillons générés (sera porté à 50 000 à l'étape 4)
theta0_max_deg = 180     # theta0 est tiré uniformément dans [0, theta0_max_deg] degrés

# === Graines (reproductibilité) ===
np.random.seed(46)       # même graine que le professeur : même tirage des architectures et de theta0


# ===== BLOC COMMUN : parametres_physiques — identique dans dataset_FM.py, test_model_FM.py — ne pas modifier ici seul =====
# Paramètres physiques du réseau (identiques à ceux du professeur)
carrierFreq = 2.45e9          # fréquence porteuse (Hz)
c = 3e8                       # vitesse de la lumière (m/s)
lambda_ = c / carrierFreq     # longueur d'onde (m)
r0 = 0.2 * lambda_            # rayon du premier anneau
delta_r = 0.5 * lambda_       # écart entre deux anneaux successifs
max_rings = 5                 # nombre d'anneaux
max_elements = 10             # nombre maximal d'éléments par anneau
# ===== FIN BLOC COMMUN =====


# ===== BLOC COMMUN : calcul_AF_performance_metrics — identique dans dataset_FM.py, test_model_FM.py — ne pas modifier ici seul =====
# Constantes et fonction du professeur (prof/test_model_FFNN.py), copiées sans modification.
# La fonction lit la variable globale theta0 (en radians) : il faut l'affecter avant chaque appel.
# Elle renvoie : AF_dB_az (diagramme normalisé en dB), HPBW (°), gain du lobe principal (dB), SSL (dB).
k = 2 * np.pi / lambda_
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)

def calcul_AF_performance_metrics(elements_per_ring):
    rings = len(elements_per_ring)
    radii = r0 + delta_r * np.arange(max_rings)
    
    AF_az = np.zeros_like(theta, dtype=complex)
    
    for ring in range(rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)
    
    
    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)
    
    #calcul du gain max non normalisé du lobe principale
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
    
    # --- HPBW ---
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3#
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))# 
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))# 
    maxIdx_ext = maxIdx + len(theta_deg)#
    

    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None #
    
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    
        
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
    
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    # --- Gain lobe principal & SSL ---
    responseLin = AF_norm_az#
    peaks, _ = find_peaks(responseLin, distance=5)
    
    pk = responseLin[peaks]#
    if len(pk) == 0:#
        main_lobe_gain = maxVal_non_norm 
        true_SSL_gain = 0# 
    else: 
        sorted_idx = np.argsort(pk)[::-1]#
        sorted_pk = pk[sorted_idx]#
        sorted_angles = theta_deg[peaks][sorted_idx]#

        threshold_dB = 1#
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]#
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)#
        
        main_lobe_gain = maxVal_non_norm
        
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
        
    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain
# ===== FIN BLOC COMMUN =====



# === Initialisation des matrices ===
Moutput = np.zeros((max_rings, nb_samples))  # Architecture [N1..N5]
Minput = np.zeros((4, nb_samples))           # [MainLobe, SSL, HPBW, theta0]

pas_progression = max(1, nb_samples // 10)   # affichage de l'avancement tous les 10 %
start_time = time.time()

print(f"Génération de {nb_samples} échantillons...")

# === Boucle principale ===
for sample_idx in range(nb_samples):
    # --- Architecture complètement aléatoire (au moins un anneau actif) ---
    # Même ordre de tirage que le professeur : d'abord l'architecture, ensuite theta0.
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:  # S'assurer qu'au moins un ring est actif
            break
    Moutput[:, sample_idx] = elements_per_ring

    # --- Theta0 aléatoire ---
    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)   # variable globale lue par calcul_AF_performance_metrics

    # --- Diagramme et performances (fonction du professeur) ---
    AF_dB_az, HPBW, main_lobe_gain, true_SSL_gain = calcul_AF_performance_metrics(elements_per_ring)

    # --- Stockage ---
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg

    # --- Avancement ---
    if (sample_idx + 1) % pas_progression == 0 or sample_idx + 1 == nb_samples:
        pourcentage = 100 * (sample_idx + 1) / nb_samples
        print(f"  {pourcentage:5.1f} %  ({sample_idx + 1}/{nb_samples} échantillons)  -  {time.time() - start_time:.1f} s")

print(f"\n=== Génération terminée en {time.time() - start_time:.2f} secondes ===")


# === Affichage d'une partie du Minput et Moutput ===
print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))


# === Résumé du jeu de données ===
print("\n=== Résumé du jeu de données ===")
noms_Minput = ["G (gain du lobe principal, dB)", "SSL (lobe secondaire, dB)", "HPBW (degrés)", "theta0 (degrés)"]
print("Plage de chaque ligne de Minput :")
for i, nom in enumerate(noms_Minput):
    print(f"  [{i}] {nom:32s} : min = {Minput[i].min():9.4f}   max = {Minput[i].max():9.4f}")

# Nombre total d'éléments N_tot = N1 + ... + N5 de chaque échantillon (entre 1 et 50)
N_tot = Moutput.sum(axis=0).astype(int)
valeurs, effectifs = np.unique(N_tot, return_counts=True)
print(f"\nNombre d'échantillons par nombre total d'éléments N_tot (min = {N_tot.min()}, max = {N_tot.max()}) :")
ligne = ""
for j, (v, n) in enumerate(zip(valeurs, effectifs)):
    ligne += f"  N_tot={v:2d} : {n:5d}"
    if (j + 1) % 5 == 0:
        print(ligne)
        ligne = ""
if ligne:
    print(ligne)


# === Enregistrement de Minput et Moutput en format .mat dans le dossier contenant le code ===
script_dir = os.path.dirname(os.path.abspath(__file__))

scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})

# === Enregistrement de Minput et Moutput en format .npy dans le dossier contenant le code ===
np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)

print("\nMinput et Moutput ont été enregistrés (.npy et .mat) dans le dossier du script.")
