# -*- coding: utf-8 -*-
r"""
outils/afficher_fichier.py : affiche le contenu des fichiers binaires (outil de développement, NON livré).

Les fichiers .npy, .mat, .npz et .pth sont binaires : VS Code ne peut pas les ouvrir comme du texte.
Ce script les lit et les affiche de façon lisible.

Utilisation (depuis la racine du dépôt) :
    .venv\Scripts\python.exe outils\afficher_fichier.py
        -> liste tous les fichiers binaires de livrable/ et prof/ avec leurs formes
    .venv\Scripts\python.exe outils\afficher_fichier.py livrable\Minput.npy
        -> forme, min/max/moyenne de chaque ligne, et les 10 premières colonnes
    .venv\Scripts\python.exe outils\afficher_fichier.py livrable\Minput.npy --colonnes 25
        -> les 25 premières colonnes
    .venv\Scripts\python.exe outils\afficher_fichier.py livrable\Moutput_test.mat --tout
        -> toutes les colonnes
    .venv\Scripts\python.exe outils\afficher_fichier.py prof\ffnn_model.pth
        -> les poids du modèle : nom, forme et nombre de paramètres de chaque couche

Formats pris en charge : .npy (numpy), .mat (MATLAB), .npz (archive numpy), .pth (poids PyTorch).
"""
import argparse
import glob
import os
import sys
import numpy as np

racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSIONS = ('.npy', '.mat', '.npz', '.pth')

# Noms des lignes des matrices du projet (CLAUDE.md §4), pour un affichage plus parlant
NOMS_LIGNES = {
    'Minput': ['G (dB)', 'SSL (dB)', 'HPBW (°)', 'theta0 (°)'],
    'Moutput': ['N1', 'N2', 'N3', 'N4', 'N5'],
}

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def noms_des_lignes(nom, nb_lignes):
    """Renvoie des étiquettes de lignes si la matrice est un Minput* ou Moutput* du projet."""
    for prefixe, noms in NOMS_LIGNES.items():
        if nom.startswith(prefixe) and len(noms) == nb_lignes:
            return noms
    return [f'ligne {i}' for i in range(nb_lignes)]


def afficher_tableau(nom, a, nb_colonnes):
    """Affiche un tableau numpy : forme, statistiques par ligne, premières colonnes."""
    a = np.asarray(a)
    print(f"\n--- {nom} : forme {a.shape}, type {a.dtype}")
    if a.size == 0:
        print("  (vide)")
        return
    if a.ndim == 0 or a.dtype.kind not in 'biuf':
        print(f"  valeur : {a}")
        return
    if a.ndim == 1:
        print(f"  min = {a.min():.4f}   max = {a.max():.4f}   moyenne = {a.mean():.4f}")
        print(f"  valeurs : {a[:nb_colonnes]}" + ("  ..." if a.size > nb_colonnes else ""))
        return
    if a.ndim == 2:
        etiquettes = noms_des_lignes(nom, a.shape[0])
        largeur = max(len(e) for e in etiquettes)
        print("  Statistiques par ligne :")
        for e, ligne in zip(etiquettes, a):
            print(f"    {e:>{largeur}} : min = {ligne.min():10.4f}   max = {ligne.max():10.4f}   "
                  f"moyenne = {ligne.mean():10.4f}")
        n = min(nb_colonnes, a.shape[1])
        entiers = bool(np.all(a == np.round(a)))   # ex. Moutput : entiers stockés en flottants
        print(f"  {n} premières colonnes (sur {a.shape[1]}) :")
        for e, ligne in zip(etiquettes, a[:, :n]):
            valeurs = ' '.join(f'{int(v):4d}' if entiers else f'{v:9.4f}' for v in ligne)
            print(f"    {e:>{largeur}} : {valeurs}")
        return
    print(f"  (tableau à {a.ndim} dimensions) premières valeurs : {a.ravel()[:nb_colonnes]}")


def lire(chemin):
    """Lit un fichier binaire et renvoie un dictionnaire {nom: tableau numpy}."""
    ext = os.path.splitext(chemin)[1].lower()
    base = os.path.splitext(os.path.basename(chemin))[0]
    if ext == '.npy':
        return {base: np.load(chemin)}
    if ext == '.npz':
        with np.load(chemin) as f:
            return {k: f[k] for k in f.files}
    if ext == '.mat':
        import scipy.io
        contenu = scipy.io.loadmat(chemin)
        return {k: v for k, v in contenu.items() if not k.startswith('__')}   # ignore __header__, __version__...
    if ext == '.pth':
        import torch
        etat = torch.load(chemin, map_location='cpu', weights_only=True)
        if not isinstance(etat, dict):
            raise ValueError("le fichier .pth ne contient pas un dictionnaire de poids (state_dict)")
        return {k: v.numpy() if hasattr(v, 'numpy') else np.asarray(v) for k, v in etat.items()}
    raise ValueError(f"extension non prise en charge : {ext} (attendu : {', '.join(EXTENSIONS)})")


def resoudre(chemin):
    """Accepte un chemin relatif au dossier courant ou à la racine du dépôt."""
    if os.path.exists(chemin):
        return chemin
    candidat = os.path.join(racine, chemin)
    if os.path.exists(candidat):
        return candidat
    sys.exit(f"Fichier introuvable : {chemin}")


def lister_tout():
    """Liste les fichiers binaires de livrable/ et prof/ avec la forme de leur contenu."""
    print("Fichiers binaires du projet (forme de chaque tableau) :")
    for dossier in ('livrable', 'prof'):
        fichiers = sorted(f for f in glob.glob(os.path.join(racine, dossier, '*'))
                          if f.lower().endswith(EXTENSIONS))
        print(f"\n{dossier}/")
        if not fichiers:
            print("  (aucun : lancer d'abord les scripts)")
        for f in fichiers:
            try:
                contenu = lire(f)
                if f.lower().endswith('.pth'):
                    nb = sum(v.size for v in contenu.values())
                    resume = f"{len(contenu)} tenseurs, {nb} paramètres"
                else:
                    resume = ', '.join(f"{k} {v.shape}" for k, v in contenu.items())
            except Exception as e:
                resume = f"illisible : {e}"
            print(f"  {os.path.basename(f):32s} {resume}")
    print("\nPour voir le contenu d'un fichier : outils\\afficher_fichier.py <chemin> [--colonnes N | --tout]")


def main():
    parser = argparse.ArgumentParser(description="Affiche le contenu des fichiers .npy, .mat, .npz et .pth.")
    parser.add_argument('fichier', nargs='?', help="fichier à afficher (sans argument : liste tous les fichiers)")
    parser.add_argument('--colonnes', type=int, default=10, help="nombre de colonnes à afficher (défaut 10)")
    parser.add_argument('--tout', action='store_true', help="afficher toutes les colonnes")
    args = parser.parse_args()

    if args.fichier is None:
        lister_tout()
        return

    chemin = resoudre(args.fichier)
    print(f"Fichier : {os.path.relpath(chemin, racine)}")
    contenu = lire(chemin)
    if chemin.lower().endswith('.pth'):
        total = 0
        print(f"\nPoids du modèle ({len(contenu)} tenseurs) :")
        for k, v in contenu.items():
            total += v.size
            print(f"  {k:30s} forme {str(v.shape):14s} {v.size:8d} paramètres   "
                  f"min = {v.min():8.4f}   max = {v.max():8.4f}")
        print(f"\nNombre total de paramètres : {total}")
        print("(pour voir les valeurs d'un tenseur, ajouter --tout)" if not args.tout else "")
        if args.tout:
            for k, v in contenu.items():
                print(f"\n--- {k}\n{v}")
        return
    nb_colonnes = sys.maxsize if args.tout else args.colonnes
    for nom, tableau in contenu.items():
        afficher_tableau(nom, tableau, nb_colonnes)


if __name__ == '__main__':
    main()
