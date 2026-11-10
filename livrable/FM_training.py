# -*- coding: utf-8 -*-
# =============================================================================
#  FM_training.py : entraînement du modèle de Flow Matching (script 3/4 du projet)
#
#  Le modèle apprend à transformer un bruit aléatoire en une architecture [N1..N5]
#  qui respecte un cahier des charges [G, SSL, HPBW, theta0] (rapport, sections 3 et 6).
#
#  Principe (rapport, fig. 7) : pour chaque exemple de la base,
#    1. on tire un bruit x0 ~ N(0, I) et un instant t ~ U[0, 1] ;
#    2. on place un point sur la ligne droite entre le bruit et l'architecture : x_t = (1 - t) x0 + t x1 ;
#    3. le réseau VelocityNet devine la vitesse du point, sachant x_t, t et le cahier des charges ;
#    4. on la compare à la vraie vitesse de la ligne droite, x1 - x0 (erreur quadratique moyenne) ;
#    5. l'optimiseur Adam corrige les poids du réseau.
#
#  Entrées (dossier du script) : Minput_training.npy (4 x N), Moutput_training.npy (5 x N)
#  Sorties (dossier du script) :
#    fm_model.pth                 poids du réseau (meilleure époque sur la validation)
#    fm_normalization_params.npz  moyenne / écart-type de la condition, n_max (mise à l'échelle)
#    fm_config.json               tous les réglages ci-dessous + l'historique des pertes
#    fm_courbe_loss.png           courbe de perte (entraînement et validation)
#
#  Version « brute » (étape 2) : sans guidage, sans appariement optimal, sans EMA.
# =============================================================================
import os
import json
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "="*80 + "\n" + " "*30 + "NOUVELLE EXÉCUTION\n" + "="*80 + "\n")

# Appel immédiat au début du script
clear_console()


# =============================================================================
# === HYPERPARAMÈTRES : tous les réglages sont ici (et enregistrés dans fm_config.json) ===
# =============================================================================
config = {
    # --- Données ---
    'donnees': {
        'proportion_validation': 0.10,       # part de l'entraînement mise de côté pour la validation
        'n_max': 10,                         # nombre max d'éléments par anneau : x = 2*N/n_max - 1 dans [-1, 1]
        'amplitude_dequantification': 0.1,   # bruit uniforme U(-0.1, 0.1) ajouté à x pendant l'entraînement seulement
    },
    # --- Réseau de vitesse VelocityNet (rapport, fig. 5-6, tableau 2) ---
    'modele': {
        'dim_x': 5,                          # 5 anneaux : N1..N5
        'dim_condition': 4,                  # condition : [G, SSL, HPBW, sin(theta0)]
        'largeur': 256,                      # projection d'entrée 5 -> 256, largeur des blocs
        'nb_blocs': 4,                       # nombre de blocs résiduels
        'nb_frequences_temps': 32,           # 32 fréquences -> 64 sinusoïdes (sin et cos)
        'freq_temps_min': 1.0,               # fréquence la plus lente (rad par unité de t)
        'freq_temps_max': 1000.0,            # fréquence la plus rapide
        'dim_temps': 64,                     # couche dense du codage du temps : 64 -> 64
        'dim_condition_cachee': 64,          # MLP de la condition : 4 -> 64 -> 64
    },
    # --- Entraînement (rapport, tableau 3) ---
    'entrainement': {
        'taux_apprentissage': 1e-3,          # Adam, taux initial
        'decroissance_cosinus': True,        # le taux décroît en cosinus jusqu'à 0 sur toutes les époques
        'taille_lot': 256,                   # nombre d'exemples par lot
        'nb_epoques': 200,                   # nombre de passages sur la base d'entraînement
        'graine': 46,                        # graine NumPy et PyTorch (reproductibilité)
        'affichage_toutes_les': 10,          # une ligne d'affichage toutes les 10 époques
    },
    # --- Améliorations prévues à l'étape 4 : DÉSACTIVÉES dans cette version brute ---
    'ameliorations': {
        'guidage_p_masque': 0.0,             # probabilité de masquer la condition (guidage sans classifieur)
        'appariement_optimal': False,        # appariement bruit / donnée par transport optimal dans chaque lot
        'ema': False,                        # moyenne glissante des poids
        'ema_decroissance': 0.999,           # facteur de l'EMA (utilisé seulement si ema = True)
    },
}

D = config['donnees']
E = config['entrainement']
A = config['ameliorations']

# === Graines (reproductibilité) ===
np.random.seed(E['graine'])
torch.manual_seed(E['graine'])

if A['appariement_optimal'] or A['ema']:
    raise NotImplementedError("L'appariement optimal et l'EMA seront ajoutés à l'étape 4.")


# =============================================================================
# === RÉSEAU DE VITESSE ===
# =============================================================================
# ===== BLOC COMMUN : VelocityNet — identique dans FM_training.py, test_model_FM.py — ne pas modifier ici seul =====
class BlocResiduelFiLM(nn.Module):
    """Un bloc de calcul (rapport, fig. 6) :
    LayerNorm -> FiLM (facteur et décalage calculés à partir du contexte) -> dense -> SiLU -> dense,
    puis on ajoute le résultat à l'entrée (raccourci) : le bloc ne calcule qu'une correction."""

    def __init__(self, largeur, dim_contexte):
        super().__init__()
        self.norm = nn.LayerNorm(largeur)                    # 1. même échelle
        self.film = nn.Linear(dim_contexte, 2 * largeur)     # 2. 256 facteurs + 256 décalages depuis le contexte
        self.dense1 = nn.Linear(largeur, largeur)            # 3. calcul de la correction
        self.activation = nn.SiLU()
        self.dense2 = nn.Linear(largeur, largeur)

    def forward(self, h, contexte):
        gamma, beta = self.film(contexte).chunk(2, dim=-1)
        u = self.norm(h)
        u = u * (1 + gamma) + beta      # FiLM ; facteur écrit 1 + gamma pour partir de l'identité à l'initialisation
        u = self.dense2(self.activation(self.dense1(u)))
        return h + u                    # 4. raccourci : entrée + correction


class VelocityNet(nn.Module):
    """Réseau de vitesse v_theta(x_t, t, c') du Flow Matching (rapport, fig. 5, tableau 2).
    Entrées : x (lot x 5) position actuelle, t (lot) instant dans [0, 1], c (lot x 4) condition centrée-réduite.
    masque (lot, booléens, optionnel) : True -> la condition est remplacée par le vecteur nul appris (guidage).
    Sortie : la vitesse prédite (lot x 5), une composante par anneau."""

    def __init__(self, dim_x, dim_condition, largeur, nb_blocs, nb_frequences_temps,
                 freq_temps_min, freq_temps_max, dim_temps, dim_condition_cachee):
        super().__init__()
        # Agrandir : 5 -> 256
        self.projection_entree = nn.Linear(dim_x, largeur)
        # Codage du temps : 32 fréquences du lent au rapide -> 64 valeurs sin/cos -> dense 64 -> 64
        frequences = torch.exp(torch.linspace(math.log(freq_temps_min), math.log(freq_temps_max),
                                              nb_frequences_temps))
        self.register_buffer('frequences', frequences, persistent=False)
        self.dense_temps = nn.Linear(2 * nb_frequences_temps, dim_temps)
        # Mise en forme de la condition : 4 -> 64 -> 64, et vecteur « vide » appris pour le guidage
        self.mlp_condition = nn.Sequential(
            nn.Linear(dim_condition, dim_condition_cachee),
            nn.SiLU(),
            nn.Linear(dim_condition_cachee, dim_condition_cachee),
        )
        self.vecteur_nul = nn.Parameter(torch.zeros(dim_condition_cachee))
        # Cœur du réseau : blocs résiduels pilotés par le contexte (temps + condition = 128)
        dim_contexte = dim_temps + dim_condition_cachee
        self.blocs = nn.ModuleList([BlocResiduelFiLM(largeur, dim_contexte) for _ in range(nb_blocs)])
        # Réduire : normalisation puis dense 256 -> 5, sans activation
        self.norm_sortie = nn.LayerNorm(largeur)
        self.dense_sortie = nn.Linear(largeur, dim_x)

    def forward(self, x, t, c, masque=None):
        angles = t.reshape(-1, 1) * self.frequences                          # (lot, 32)
        code_temps = self.dense_temps(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        code_condition = self.mlp_condition(c)
        if masque is not None:
            code_condition = torch.where(masque.reshape(-1, 1),
                                         self.vecteur_nul.expand_as(code_condition), code_condition)
        contexte = torch.cat([code_temps, code_condition], dim=-1)           # (lot, 128)
        h = self.projection_entree(x)
        for bloc in self.blocs:
            h = bloc(h, contexte)
        return self.dense_sortie(self.norm_sortie(h))
# ===== FIN BLOC COMMUN =====


# =============================================================================
# === DONNÉES ===
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# Les fichiers stockent caractéristiques x échantillons : on transpose en échantillons x caractéristiques
Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy')).T    # (N, 4) : G, SSL, HPBW, theta0
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy')).T  # (N, 5) : N1..N5

# --- Mise de côté de 10 % pour la validation (tirage avec la graine 46) ---
nb_total = Minput_training.shape[0]
permutation = np.random.permutation(nb_total)
nb_val = round(D['proportion_validation'] * nb_total)
idx_val = permutation[:nb_val]
idx_train = permutation[nb_val:]
nb_train = len(idx_train)

# --- Condition c' = [G, SSL, HPBW, sin(theta0)], centrée-réduite avec la partie entraînement seulement ---
# On donne sin(theta0) car theta0 et 180° - theta0 donnent le même diagramme (propriété P2).
condition_brute = np.column_stack([Minput_training[:, 0], Minput_training[:, 1], Minput_training[:, 2],
                                   np.sin(np.deg2rad(Minput_training[:, 3]))])
cond_mean = condition_brute[idx_train].mean(axis=0)
cond_std = condition_brute[idx_train].std(axis=0)
condition = (condition_brute - cond_mean) / cond_std

# --- Architecture ramenée dans [-1, 1] : x = 2*N/n_max - 1 (même échelle que le bruit de départ) ---
x_architecture = 2 * Moutput_training / D['n_max'] - 1

x1_train = torch.tensor(x_architecture[idx_train], dtype=torch.float32)
c_train = torch.tensor(condition[idx_train], dtype=torch.float32)
x1_val = torch.tensor(x_architecture[idx_val], dtype=torch.float32)
c_val = torch.tensor(condition[idx_val], dtype=torch.float32)

# La perte du Flow Matching dépend du bruit x0 et de l'instant t tirés au hasard. Pour que la perte de
# validation soit comparable d'une époque à l'autre, on tire x0 et t UNE SEULE FOIS pour la validation.
generateur_val = torch.Generator().manual_seed(E['graine'] + 1)
x0_val = torch.randn(x1_val.shape, generator=generateur_val)
t_val = torch.rand(nb_val, generator=generateur_val)

print(f"Données : {nb_total} exemples d'entraînement -> {nb_train} pour apprendre, {nb_val} pour la validation")
print(f"Condition : moyenne = {np.round(cond_mean, 4)}   écart-type = {np.round(cond_std, 4)}")


# =============================================================================
# === MODÈLE, OPTIMISEUR ===
# =============================================================================
model = VelocityNet(**config['modele'])
nb_parametres = sum(p.numel() for p in model.parameters())
print(f"\nRéseau VelocityNet : {nb_parametres} paramètres ({nb_parametres * 4 / 1e6:.1f} Mo en float32)")

optimizer = torch.optim.Adam(model.parameters(), lr=E['taux_apprentissage'])
scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=E['nb_epoques'])
             if E['decroissance_cosinus'] else None)


def perte_flow_matching(x1, c, x0, t, masque=None):
    """Erreur quadratique moyenne entre la vitesse prédite et la vitesse de la ligne droite x1 - x0."""
    xt = (1 - t.reshape(-1, 1)) * x0 + t.reshape(-1, 1) * x1      # point sur la ligne droite (éq. 2)
    v_predite = model(xt, t, c, masque)
    return ((v_predite - (x1 - x0)) ** 2).mean()                   # éq. 3


# =============================================================================
# === ENTRAÎNEMENT (rapport, algorithme 1 et fig. 7) ===
# =============================================================================
loss_train_liste, loss_val_liste = [], []
meilleure_loss_val = float('inf')
meilleure_epoque = 0
meilleurs_poids = None

print(f"\nEntraînement : {E['nb_epoques']} époques, lots de {E['taille_lot']}, Adam lr = {E['taux_apprentissage']}"
      f"{' (décroissance en cosinus)' if E['decroissance_cosinus'] else ''}")
start_time = time.time()

for epoque in range(1, E['nb_epoques'] + 1):
    model.train()
    ordre = torch.randperm(nb_train)          # on mélange les exemples à chaque époque
    somme_pertes = 0.0
    for debut in range(0, nb_train, E['taille_lot']):
        idx = ordre[debut:debut + E['taille_lot']]
        n = len(idx)
        # Un vrai exemple, déquantifié (bruit uniforme ±0.1 pour rendre les entiers continus)
        x1 = x1_train[idx]
        x1 = x1 + (2 * torch.rand_like(x1) - 1) * D['amplitude_dequantification']
        c = c_train[idx]
        x0 = torch.randn_like(x1)             # un départ au hasard
        t = torch.rand(n)                     # un instant au hasard
        masque = torch.rand(n) < A['guidage_p_masque']   # condition masquée (jamais si p = 0)
        loss = perte_flow_matching(x1, c, x0, t, masque)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        somme_pertes += loss.item() * n
    if scheduler is not None:
        scheduler.step()
    loss_train = somme_pertes / nb_train      # moyenne par exemple (et non somme, contrairement au FFNN du cours)

    # --- Validation : mêmes x0 et t à chaque époque, sans déquantification ---
    model.eval()
    with torch.no_grad():
        loss_val = perte_flow_matching(x1_val, c_val, x0_val, t_val).item()

    loss_train_liste.append(loss_train)
    loss_val_liste.append(loss_val)
    if loss_val < meilleure_loss_val:         # on garde les poids de la meilleure époque
        meilleure_loss_val = loss_val
        meilleure_epoque = epoque
        meilleurs_poids = copy.deepcopy(model.state_dict())

    if epoque == 1 or epoque % E['affichage_toutes_les'] == 0:
        print(f"Époque {epoque:4d}/{E['nb_epoques']}  loss train = {loss_train:.4f}  loss val = {loss_val:.4f}  "
              f"lr = {optimizer.param_groups[0]['lr']:.2e}")

duree = time.time() - start_time
print(f"\n=== Entraînement terminé en {duree:.2f} secondes ===")
print(f"Meilleure époque (validation) : {meilleure_epoque}   loss val = {meilleure_loss_val:.4f}")
print(f"Loss train : première = {loss_train_liste[0]:.4f}   dernière = {loss_train_liste[-1]:.4f}")
print(f"Loss val   : première = {loss_val_liste[0]:.4f}   dernière = {loss_val_liste[-1]:.4f}")


# =============================================================================
# === SAUVEGARDES (dossier du script) ===
# =============================================================================
torch.save(meilleurs_poids, os.path.join(script_dir, 'fm_model.pth'))

np.savez(os.path.join(script_dir, 'fm_normalization_params.npz'),
         cond_mean=cond_mean, cond_std=cond_std, n_max=D['n_max'])

config['resultats'] = {
    'nb_parametres': nb_parametres,
    'nb_train': nb_train,
    'nb_val': nb_val,
    'meilleure_epoque': meilleure_epoque,
    'meilleure_loss_val': meilleure_loss_val,
    'duree_entrainement_s': round(duree, 2),
    'loss_train': [round(v, 6) for v in loss_train_liste],
    'loss_val': [round(v, 6) for v in loss_val_liste],
}
with open(os.path.join(script_dir, 'fm_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# --- Courbe de perte ---
epoques = np.arange(1, E['nb_epoques'] + 1)
plt.figure(figsize=(9, 5))
plt.plot(epoques, loss_train_liste, color='#2a78d6', linewidth=2, label='Entraînement')
plt.plot(epoques, loss_val_liste, color='#eb6834', linewidth=2, label='Validation')
plt.plot(meilleure_epoque, meilleure_loss_val, 'o', color='#eb6834', markersize=8,
         markeredgecolor='white', markeredgewidth=2)
plt.annotate(f"meilleure époque : {meilleure_epoque}\nloss val = {meilleure_loss_val:.4f}",
             xy=(meilleure_epoque, meilleure_loss_val), xytext=(10, 25), textcoords='offset points',
             fontsize=9, color='#333333')
plt.xlabel("Époque")
plt.ylabel("Perte MSE (moyenne par composante)")
plt.title("Courbe de perte du Flow Matching (training loss et validation loss)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fm_courbe_loss.png'), dpi=150)
plt.show()

print("\nFichiers enregistrés dans le dossier du script : fm_model.pth, fm_normalization_params.npz, "
      "fm_config.json, fm_courbe_loss.png")
