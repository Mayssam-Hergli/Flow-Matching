# -*- coding: utf-8 -*-
# =============================================================================
#  repartition_training_test_FM.py : répartition entraînement / test (script 2/4 du projet Flow Matching)
#
#  Lit Minput.npy (4 x N) et Moutput.npy (5 x N) produits par dataset_FM.py,
#  mélange les colonnes (échantillons) avec la graine 46, puis garde :
#    - 10 % des échantillons pour le test,
#    - les 90 % restants pour l'entraînement.
#  (La validation, 10 % de l'entraînement, sera prélevée plus tard dans FM_training.py.)
#
#  Sorties (dans le dossier du script, formats .npy et .mat, clé .mat = nom de la variable) :
#    Minput_training, Moutput_training, Minput_test, Moutput_test
#
#  Comportement identique à prof/repartition_training_test.py.
# =============================================================================
import numpy as np
import scipy.io
import os


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()


# === Paramètres modifiables ===
proportion_test = 0.10   # part des échantillons réservée au test (comme le professeur)

# === Graines (reproductibilité) ===
np.random.seed(46)       # même graine que le professeur : même permutation, donc même répartition


# === Chargement des matrices Minput et Moutput (dossier qui contient le script) ===
script_dir = os.path.dirname(os.path.abspath(__file__))

Minput = np.load(os.path.join(script_dir, 'Minput.npy'))     # 4 x N : [G ; SSL ; HPBW ; theta0]
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))   # 5 x N : [N1 ; ... ; N5]


# === Mélange aléatoire des indices des échantillons (colonnes) ===
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

# === Séparation : 10 % test / le reste training ===
nb_test = round(proportion_test * total_cols)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

# === Sélection des colonnes ===
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]


# === Affichage d'une partie du Minput_training et Moutput_training ===
print("\nCaractéristiques de la matrice Minput_training :")
print(f"- Shape (lignes, colonnes) : {Minput_training.shape}")
print("Premières colonnes (extraits) :")
print(np.round(Minput_training[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput_training :")
print(f"- Shape (lignes, colonnes) : {Moutput_training.shape}")
print("Premières colonnes (extraits) :")
print(Moutput_training[:, :5].astype(int))


# === Affichage d'une partie du Minput_test et Moutput_test ===
print("\nCaractéristiques de la matrice Minput_test :")
print(f"- Shape (lignes, colonnes) : {Minput_test.shape}")
print("Premières colonnes (extraits) :")
print(np.round(Minput_test[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput_test :")
print(f"- Shape (lignes, colonnes) : {Moutput_test.shape}")
print("Premières colonnes (extraits) :")
print(Moutput_test[:, :5].astype(int))


# === Enregistrement au format .mat (clé = nom de la variable) ===
scipy.io.savemat(os.path.join(script_dir, 'Minput_training.mat'), {'Minput_training': Minput_training})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_training.mat'), {'Moutput_training': Moutput_training})
scipy.io.savemat(os.path.join(script_dir, 'Minput_test.mat'), {'Minput_test': Minput_test})
scipy.io.savemat(os.path.join(script_dir, 'Moutput_test.mat'), {'Moutput_test': Moutput_test})

# === Enregistrement au format .npy ===
np.save(os.path.join(script_dir, 'Minput_training.npy'), Minput_training)
np.save(os.path.join(script_dir, 'Moutput_training.npy'), Moutput_training)
np.save(os.path.join(script_dir, 'Minput_test.npy'), Minput_test)
np.save(os.path.join(script_dir, 'Moutput_test.npy'), Moutput_test)

print("\nLes 4 matrices training / test ont été enregistrées (.npy et .mat) dans le dossier du script.")
