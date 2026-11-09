# -*- coding: utf-8 -*-
# =============================================================================
#  outils/verifier_blocs_communs.py : vérifie les BLOCS COMMUNS (outil de développement, NON livré)
#
#  Règle (CLAUDE.md §6) : le code partagé entre les scripts de livrable/ est COPIÉ, pas importé.
#  Chaque copie est encadrée par :
#    # ===== BLOC COMMUN : <nom> — identique dans <scripts> — ne pas modifier ici seul =====
#    ...
#    # ===== FIN BLOC COMMUN =====
#
#  Ce script :
#    1. trouve tous les blocs dans livrable/*.py et les regroupe par nom ;
#    2. vérifie que toutes les copies d'un même bloc sont identiques (en-tête compris) ;
#    3. vérifie que chaque script listé dans l'en-tête, s'il existe déjà, contient bien le bloc ;
#    4. vérifie que la fonction calcul_AF_performance_metrics copiée est identique à celle de
#       prof/test_model_FFNN.py (règle 2 du professeur : copie mot pour mot).
#  Les fins de ligne (CRLF / LF) sont ignorées dans les comparaisons.
#
#  Lancement (depuis la racine du dépôt) : .venv\Scripts\python.exe outils\verifier_blocs_communs.py
#  Code de sortie 0 si tout passe, 1 sinon.
# =============================================================================
import ast
import difflib
import glob
import os
import re
import sys

racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dossier_livrable = os.path.join(racine, 'livrable')
fichier_prof = os.path.join(racine, 'prof', 'test_model_FFNN.py')

re_debut = re.compile(r'^# ===== BLOC COMMUN : (\S+) — identique dans (.+?) — ne pas modifier ici seul =====$')
re_fin = re.compile(r'^# ===== FIN BLOC COMMUN =====$')

erreurs = []
blocs = {}       # nom -> liste de (fichier, texte du bloc, liste des scripts annoncés)

print("=" * 80)
print("VÉRIFICATION DES BLOCS COMMUNS (livrable/)")
print("=" * 80)

# --- 1. Extraction des blocs ---
for chemin in sorted(glob.glob(os.path.join(dossier_livrable, '*.py'))):
    fichier = os.path.basename(chemin)
    lignes = open(chemin, encoding='utf-8').read().splitlines()
    en_cours = None   # (nom, scripts, numéro de ligne de début, lignes du bloc)
    for num, ligne in enumerate(lignes, start=1):
        m = re_debut.match(ligne)
        if 'BLOC COMMUN' in ligne and not m and not re_fin.match(ligne):
            erreurs.append(f"{fichier}:{num} : marqueur BLOC COMMUN mal formé : {ligne.strip()}")
            continue
        if m:
            if en_cours is not None:
                erreurs.append(f"{fichier}:{num} : bloc '{m.group(1)}' ouvert avant la fin du bloc '{en_cours[0]}'")
            scripts = [s.strip() for s in m.group(2).split(',')]
            en_cours = (m.group(1), scripts, num, [ligne])
        elif re_fin.match(ligne):
            if en_cours is None:
                erreurs.append(f"{fichier}:{num} : FIN BLOC COMMUN sans début")
                continue
            nom, scripts, debut, contenu = en_cours
            contenu.append(ligne)
            blocs.setdefault(nom, []).append((fichier, '\n'.join(contenu), scripts))
            print(f"  trouvé : bloc '{nom}' dans {fichier} (lignes {debut}-{num})")
            en_cours = None
        elif en_cours is not None:
            en_cours[3].append(ligne)
    if en_cours is not None:
        erreurs.append(f"{fichier}:{en_cours[2]} : bloc '{en_cours[0]}' jamais fermé")

if not blocs:
    erreurs.append("aucun bloc commun trouvé dans livrable/")

# --- 2. et 3. Comparaison des copies et présence dans les scripts annoncés ---
for nom, copies in sorted(blocs.items()):
    fichiers = [f for f, _, _ in copies]
    reference_fichier, reference_texte, scripts_annonces = copies[0]
    identiques = True
    for fichier, texte, _ in copies[1:]:
        if texte != reference_texte:
            identiques = False
            diff = difflib.unified_diff(reference_texte.splitlines(), texte.splitlines(),
                                        reference_fichier, fichier, lineterm='', n=1)
            print('\n'.join(list(diff)[:30]))
    manquants = [s for s in scripts_annonces
                 if os.path.exists(os.path.join(dossier_livrable, s)) and s not in fichiers]
    pas_encore_crees = [s for s in scripts_annonces if not os.path.exists(os.path.join(dossier_livrable, s))]
    ok = identiques and not manquants
    print(f"\n[{'PASS' if ok else 'FAIL'}] bloc '{nom}' : {len(copies)} copie(s) dans {', '.join(fichiers)}")
    if len(copies) == 1:
        print("        une seule copie pour l'instant : rien à comparer")
    elif identiques:
        print("        toutes les copies sont identiques")
    else:
        erreurs.append(f"bloc '{nom}' : copies différentes")
    if manquants:
        erreurs.append(f"bloc '{nom}' : absent de {', '.join(manquants)} alors que l'en-tête l'annonce")
        print(f"        ABSENT de : {', '.join(manquants)}")
    if pas_encore_crees:
        print(f"        scripts annoncés pas encore créés : {', '.join(pas_encore_crees)}")

# --- 4. La fonction copiée est-elle celle du professeur ? ---
print("\nComparaison avec la fonction originale du professeur (prof/test_model_FFNN.py) :")


def extraire_fonction(source, nom='calcul_AF_performance_metrics'):
    for noeud in ast.parse(source).body:
        if isinstance(noeud, ast.FunctionDef) and noeud.name == nom:
            return ast.get_source_segment(source, noeud)
    return None


fonction_prof = extraire_fonction(open(fichier_prof, encoding='utf-8').read())
copies_fonction = blocs.get('calcul_AF_performance_metrics', [])
if not copies_fonction:
    erreurs.append("bloc 'calcul_AF_performance_metrics' introuvable")
    print("  [FAIL] bloc 'calcul_AF_performance_metrics' introuvable")
for fichier, texte, _ in copies_fonction:
    fonction_copiee = extraire_fonction(texte)
    ok = fonction_copiee is not None and fonction_copiee == fonction_prof
    print(f"  [{'PASS' if ok else 'FAIL'}] {fichier} : fonction identique à l'originale")
    if not ok:
        erreurs.append(f"{fichier} : calcul_AF_performance_metrics différente de celle du professeur")

# --- Bilan ---
print("\n" + "=" * 80)
if erreurs:
    print("ÉCHEC :")
    for e in erreurs:
        print(f"  - {e}")
else:
    print("TOUS LES BLOCS COMMUNS SONT COHÉRENTS")
print("=" * 80)
sys.exit(1 if erreurs else 0)
