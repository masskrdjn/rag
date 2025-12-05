#!/usr/bin/env python3
"""
Script pour vider le cache du RAG.
Le cache est stocké dans le dossier .cache/ du projet.
"""
import os
import glob
import shutil

# Chemin du cache (relatif au dossier du projet)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

def clear_cache():
    """Supprime tous les fichiers de cache"""
    if not os.path.exists(CACHE_DIR):
        print(f"❌ Dossier de cache non trouvé: {CACHE_DIR}")
        return 0
    
    # Compter les fichiers
    files = glob.glob(f"{CACHE_DIR}/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    count = len(files)
    
    if count == 0:
        print("✓ Cache déjà vide")
        return 0
    
    # Supprimer tout le contenu du dossier .cache
    for item in os.listdir(CACHE_DIR):
        item_path = os.path.join(CACHE_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    print(f"✓ {count} fichiers de cache supprimés")
    return count

if __name__ == "__main__":
    clear_cache()