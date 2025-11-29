# Guide Complet : Ingestion de Fichiers HTML dans le Système RAG

## Vue d'ensemble

Ce guide décrit la procédure complète pour ingérer des fichiers HTML dans votre système RAG avec extraction automatique des URLs d'images et chunking optimisé pour les documents WordPress.

> [!IMPORTANT]
> **Version Optimisée** : Ce guide utilise `ingest_html_optimized.py` qui intègre :
> - ✅ Extraction automatique des URLs d'images (avec BeautifulSoup)
> - ✅ Séparateurs personnalisés pour listes et titres
> - ✅ Chunking optimisé (800 chars / 100 overlap)
> - ✅ Support des procédures WordPress avec captures d'écran

---

## 🚀 Méthode Rapide (Recommandée)

### Workflow Automatisé en Une Commande

```bash
# Rendre le script exécutable (une seule fois)
chmod +x /home/rag/ingest_html_optimized_workflow.sh

# Lancer l'ingestion complète
sudo bash /home/rag/ingest_html_optimized_workflow.sh /chemin/vers/vos/fichiers/html/
```

**Ce script fait tout automatiquement** :
1. Copie vos fichiers HTML
2. Installe les dépendances (`unstructured` + `beautifulsoup4`)
3. Chunke et indexe les documents avec extraction d'images
4. Redémarre le serveur RAG
5. Vérifie que tout fonctionne

---

## 📋 Méthode Manuelle (Étape par Étape)

Si vous préférez contrôler chaque étape :

### Étape 1 : Copier les Fichiers HTML

```bash
# Copier vos fichiers HTML dans le répertoire data
sudo cp /chemin/source/*.html /home/ragapp/rag-system/data/
sudo chown ragapp:ragapp /home/ragapp/rag-system/data/*.html

# Vérifier les fichiers copiés
ls -lh /home/ragapp/rag-system/data/
```

> [!TIP]
> Vous pouvez organiser vos HTML dans des sous-dossiers par thématique. Le script les trouvera automatiquement.

---

### Étape 2 : Installer les Dépendances

```bash
# Se connecter en tant que ragapp et installer les dépendances
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && source venv/bin/activate && pip install unstructured[html] beautifulsoup4 && deactivate"
```

**Dépendances installées** :
- `unstructured[html]` : Parser HTML
- `beautifulsoup4` : Extraire les URLs d'images

**Note** : Installation peut prendre 2-3 minutes.

---

### Étape 3 : Exécuter l'Ingestion Optimisée

```bash
# Lancer le script d'ingestion optimisé
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest_html_optimized.py"
```

**Sortie attendue** :
```
Loading HTML documents from data...
Found X HTML file(s)
✓ Loaded: article1.html
✓ Loaded: article2.html
...
✓ Loaded X HTML document(s)
Total characters: XXXXX
✓ Split into XX chunks
Average chunk size: XXX characters

SAMPLE CHUNK (first one):
=========================================
Source: /home/ragapp/rag-system/data/article1.html
Content preview (first 300 chars):
[Texte de l'article...]

=== IMAGES RÉFÉRENCÉES ===
Image 1: Capture écran du tableau de bord
URL: https://example.com/uploads/dashboard.png
...
=========================================

✓ INGESTION COMPLETE!
=========================================
📊 Summary:
  - Documents processed: X
  - Chunks created: XX
  - Average chunk size: XXX chars
  - Embedding model: nomic-embed-text
  - Database location: chroma_db

💡 Tips:
  - Images URLs are now indexed and searchable
  - Chunks are optimized for procedural content
  - Don't forget to restart your RAG server!
```

---

### Étape 4 : Redémarrer le Serveur

```bash
# Arrêter le serveur actuel
sudo pkill -f uvicorn

# Relancer le serveur avec les nouvelles données
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"

# Attendre le démarrage
sleep 5
```

---

### Étape 5 : Vérifier l'Indexation

```bash
# Test de santé du serveur
curl -s http://localhost:8000/health

# Test avec une question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Où se trouve l image du tableau de bord ?"}'
```

**Résultat attendu** : Le système devrait trouver les URLs d'images !

---

## ✨ Avantages de la Version Optimisée

### 1. Extraction des URLs d'Images 🖼️

**Avant** (version standard) :
- Balises `<img>` supprimées
- URLs d'images perdues ❌

**Maintenant** (version optimisée) :
- URLs extraites et ajoutées au texte
- Section `=== IMAGES RÉFÉRENCÉES ===` dans chaque document ✅

**Exemple de texte indexé** :
```
Procédure de configuration WordPress...

=== IMAGES RÉFÉRENCÉES ===

Image 1: Capture écran du tableau de bord
URL: https://example.com/wp-content/uploads/dashboard.png

Image 2: Menu de configuration
URL: https://example.com/wp-content/uploads/menu-config.png
```

### 2. Chunking Optimisé

**Paramètres** :
- `chunk_size=800` : Taille optimale pour procédures
- `chunk_overlap=100` : 12.5% overlap (meilleur ratio)

**Séparateurs personnalisés** :
```python
separators=[
    "\n\n",          # Paragraphes
    "\n- ",          # Listes à puces
    "\n* ",          # Listes alternatives
    "\n# ", "\n## ", # Titres markdown
    "\n",            # Lignes
    " ",             # Mots
]
```

**Résultat** : Chunks qui ne coupent pas au milieu d'une liste ou d'un titre !

---

## 🔄 Gestion Continue

### Ajouter de Nouveaux Fichiers HTML

```bash
# 1. Copier les nouveaux fichiers
sudo cp /nouveau/chemin/*.html /home/ragapp/rag-system/data/
sudo chown ragapp:ragapp /home/ragapp/rag-system/data/*.html

# 2. Ré-exécuter l'ingestion (méthode rapide)
sudo bash /home/rag/ingest_html_optimized_workflow.sh
```

### Repartir de Zéro

```bash
# Supprimer l'ancienne base
sudo rm -rf /home/ragapp/rag-system/chroma_db/

# Relancer l'ingestion
sudo bash /home/rag/ingest_html_optimized_workflow.sh /chemin/vers/vos/html
```

---

## 🛠️ Troubleshooting

### Problème : "No module named 'bs4'"
**Cause** : BeautifulSoup4 non installé

**Solution** :
```bash
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && source venv/bin/activate && pip install beautifulsoup4 && deactivate"
```

### Problème : "No module named 'unstructured'"
**Cause** : Unstructured non installé

**Solution** :
```bash
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && source venv/bin/activate && pip install unstructured[html] && deactivate"
```

### Problème : Aucun document chargé
**Vérifications** :
```bash
# Vérifier que les fichiers HTML sont présents
ls -lh /home/ragapp/rag-system/data/*.html

# Vérifier les permissions
ls -l /home/ragapp/rag-system/data/
```

### Problème : Les images ne sont pas indexées
**Vérification** : Assurez-vous d'utiliser `ingest_html_optimized.py` et non `ingest_html.py`

```bash
# Vérifier la présence du script optimisé
ls -l /home/ragapp/rag-system/ingest_html_optimized.py
```

### Problème : Le serveur ne voit pas les nouvelles données
**Solution** : Toujours redémarrer le serveur après ingestion

```bash
sudo pkill -f uvicorn
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

---

## 📝 Résumé des Commandes

### Workflow Complet (Une Commande)
```bash
sudo bash /home/rag/ingest_html_optimized_workflow.sh /source/html
```

### Workflow Manuel
```bash
# 1. Copier
sudo cp /source/*.html /home/ragapp/rag-system/data/
sudo chown ragapp:ragapp /home/ragapp/rag-system/data/*.html

# 2. Installer dépendances (une fois)
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && source venv/bin/activate && pip install unstructured[html] beautifulsoup4 && deactivate"

# 3. Ingérer
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest_html_optimized.py"

# 4. Redémarrer
sudo pkill -f uvicorn
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"

# 5. Tester
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "Test"}'
```

---

## 📚 Documentation Complémentaire

- `COMPARAISON_VERSIONS.md` : Comparaison détaillée standard vs optimisé
- `ingest_html_optimized.py` : Code source du script d'ingestion
- `ingest_html_optimized_workflow.sh` : Script bash automatisé

---

## ℹ️ Informations Techniques

**Fichiers de configuration** :
- Répertoire data : `/home/ragapp/rag-system/data/`
- Base ChromaDB : `/home/ragapp/rag-system/chroma_db/`
- Modèle embedding : `nomic-embed-text`
- Modèle LLM : `llama3.2`

**Paramètres de chunking** :
- Taille : 800 caractères
- Overlap : 100 caractères (12.5%)
- Séparateurs : Adaptés au contenu WordPress
