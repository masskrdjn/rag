# Guide Complet : Ingestion de Fichiers HTML dans le Système RAG

## Vue d'ensemble

Ce guide décrit la procédure complète pour ingérer des fichiers HTML dans votre système RAG avec extraction automatique des URLs d'images et chunking optimisé pour les documents WordPress.

> [!IMPORTANT]
> **Version Optimisée** : Ce guide utilise `ingest_html_adaptive.py` qui intègre :
> - ✅ Extraction automatique des URLs d'images (avec BeautifulSoup)
> - ✅ Séparateurs personnalisés pour listes et titres
> - ✅ Chunking optimisé (800 chars / 100 overlap)
> - ✅ Support des procédures WordPress avec captures d'écran

---

## 🚀 Méthode Rapide (Recommandée)

### Ingestion en Une Commande

```bash
# Lancer l'ingestion complète
cd /home/rag && python3 ingest_html_adaptive.py
```

---

## 📋 Méthode Manuelle (Étape par Étape)

### Étape 1 : Copier les Fichiers HTML

```bash
# Copier vos fichiers HTML dans le répertoire data
cp /chemin/source/*.html /home/rag/data/

# Vérifier les fichiers copiés
ls -lh /home/rag/data/
```

---

### Étape 2 : Installer les Dépendances (si nécessaire)

```bash
pip3 install beautifulsoup4
```

---

### Étape 3 : Exécuter l'Ingestion

```bash
cd /home/rag && python3 ingest_html_adaptive.py
```

**Sortie attendue** :
```
Loading HTML documents from data...
Found X HTML file(s)
✓ Loaded X HTML document(s)
Total characters: XXXXX
✓ Split into XX chunks
Average chunk size: XXX characters

✓ INGESTION COMPLETE!
```

---

### Étape 4 : Redémarrer le Serveur

```bash
bash /home/rag/restart_server.sh
```

---

### Étape 5 : Vérifier l'Indexation

```bash
# Test de santé du serveur
curl -s http://localhost:8000/health

# Test avec une question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Votre question ici"}'
```

---

## 🔄 Gestion Continue

### Ajouter de Nouveaux Fichiers HTML

```bash
# 1. Copier les nouveaux fichiers
cp /nouveau/chemin/*.html /home/rag/data/

# 2. Ré-exécuter l'ingestion
cd /home/rag && python3 ingest_html_adaptive.py

# 3. Redémarrer le serveur
bash /home/rag/restart_server.sh
```

### Repartir de Zéro

```bash
# Supprimer l'ancienne base
rm -rf /home/rag/chroma_db/

# Relancer l'ingestion
cd /home/rag && python3 ingest_html_adaptive.py

# Redémarrer le serveur
bash /home/rag/restart_server.sh
```

---

## 🛠️ Troubleshooting

### Problème : "No module named 'bs4'"

**Solution** :
```bash
pip3 install beautifulsoup4
```

### Problème : Le serveur ne voit pas les nouvelles données

**Solution** : Toujours redémarrer le serveur après ingestion
```bash
bash /home/rag/restart_server.sh
```

---

## 📝 Résumé des Commandes

| Action | Commande |
|--------|----------|
| Copier fichiers | `cp /source/*.html /home/rag/data/` |
| Ingérer | `cd /home/rag && python3 ingest_html_adaptive.py` |
| Redémarrer | `bash /home/rag/restart_server.sh` |
| Tester | `curl http://localhost:8000/health` |

---

## ℹ️ Informations Techniques

**Chemins** :
- Répertoire data : `/home/rag/data/`
- Base ChromaDB : `/home/rag/chroma_db/`

**Modèles** :
- Modèle embedding : `nomic-embed-text`
- Modèle LLM : `mistral:7b`

**Paramètres de chunking** :
- Taille : 800 caractères
- Overlap : 100 caractères (12.5%)
