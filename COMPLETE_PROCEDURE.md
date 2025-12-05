# Système RAG - Procédure complète d'installation et d'exploitation

## Table des matières
1. [Prérequis](#prérequis)
2. [Étapes d'installation](#étapes-dinstallation)
3. [Configuration GPU/CPU](#configuration-gpucpu)
4. [Ingestion des données](#ingestion-des-données)
5. [Gestion des services](#gestion-des-services)
6. [Dépannage](#dépannage)
7. [Maintenance](#maintenance)

---

## Prérequis

- **OS :** Ubuntu 22.04 ou ultérieur (testé sur WSL2)
- **Privilèges :** accès sudo
- **Réseau :** connexion Internet pour télécharger les modèles (~2-4 Go)
- **Espace disque :** ~10 Go libres (pour les modèles et ChromaDB)
- **Répertoire d'installation :** `/home/rag/`
- **GPU (optionnel) :** NVIDIA avec support CUDA pour accélération

---

## Étapes d'installation

### Méthode 1 : Installation automatisée (Recommandée)

```bash
# Télécharger et exécuter le script d'installation complet
sudo bash complete_install.sh
```

Ce script va :
1. Installer les dépendances système (Python, pip, git, curl)
2. Installer Ollama
3. Configurer le répertoire `/home/rag/`
4. Télécharger les modèles IA (mistral:7b, nomic-embed-text)
5. Installer les dépendances Python
6. Ingérer les données
7. Démarrer le service

### Méthode 2 : Installation manuelle étape par étape

#### Étape 1 : Dépendances système
```bash
# Mettre à jour le système
sudo apt update && sudo apt upgrade -y

# Installer Python et les outils
sudo apt install -y python3 python3-pip python3-venv git curl

# Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### Étape 2 : Configuration du répertoire
```bash
# Créer le répertoire de l'application (si nécessaire)
mkdir -p /home/rag
mkdir -p /home/rag/data

# S'assurer que l'utilisateur courant a les droits
sudo chown -R $USER:$USER /home/rag
```

#### Étape 3 : Téléchargement des modèles IA
```bash
# Tirer le modèle LLM (pour générer des réponses)
ollama pull mistral:7b

# Tirer le modèle d'intégration (pour la vectorisation)
ollama pull nomic-embed-text
```

#### Étape 4 : Configuration de l'application
```bash
# Les fichiers de l'application sont déjà dans /home/rag/
cd /home/rag

# Vérifier que les fichiers principaux sont présents
ls -la server.py rag_pipeline.py ingest_html_adaptive.py
```

#### Étape 5 : Dépendances Python
```bash
# Installer les dépendances Python (installation globale)
pip3 install --upgrade pip
pip3 install \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-ollama \
    chromadb \
    fastapi \
    uvicorn \
    python-dotenv \
    beautifulsoup4 \
    sentence-transformers
```

---

## Configuration GPU/CPU

Le système supporte deux modes de fonctionnement :
- **Mode CPU (défaut)** : Utilise BM25 et TF-IDF pour le reranking
- **Mode GPU** : Utilise CrossEncoder et SentenceTransformer pour de meilleures performances

### Configuration via variable d'environnement

```bash
# Vérifier le mode actuel
python3 -c "from device_config import print_device_status; print_device_status()"

# Forcer mode CPU (par défaut)
export RAG_USE_GPU=0

# Forcer mode GPU
export RAG_USE_GPU=1

# Détection automatique
export RAG_USE_GPU=auto
```

### Configuration pour le service systemd

```bash
# Éditer le fichier du service
sudo nano /etc/systemd/system/rag-api.service

# Ajouter dans la section [Service] :
# Environment="RAG_USE_GPU=1"   ← pour GPU
# Environment="RAG_USE_GPU=0"   ← pour CPU (défaut)

# Recharger et redémarrer
sudo systemctl daemon-reload
sudo systemctl restart rag-api
```

### Installation GPU (si GPU NVIDIA disponible)

```bash
# Installer PyTorch avec support CUDA
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# Vérifier l'installation
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## Ingestion des données

### Ingestion initiale des données

1. **Placez vos documents** dans `/home/rag/data/` :
   ```bash
   # Copier vos fichiers HTML dans le répertoire data
   cp /chemin/vers/vos/fichiers/*.html /home/rag/data/
   ```

2. **Exécutez le script d'ingestion** :
   ```bash
   cd /home/rag && python3 ingest_html_adaptive.py
   ```

   **Sortie attendue :**
   ```
   Chargement des documents depuis data...
   Chargé X documents.
   Divisé en Y morceaux.
   Sauvegarde dans ChromaDB...
   Ingestion terminée ! Données sauvegardées dans chroma_db
   ```

### Ce qui se passe pendant l'ingestion

Le script `ingest_html_adaptive.py` effectue les opérations suivantes :

1. **Chargement :** Lit tous les fichiers `.html` du répertoire `data/`
2. **Découpage :** Divise les documents en morceaux gérables :
   - Taille des morceaux : 1000 caractères
   - Chevauchement : 200 caractères (pour la continuité du contexte)
3. **Intégration :** Convertit les morceaux de texte en vecteurs utilisant le modèle `nomic-embed-text`
4. **Stockage :** Sauvegarde les vecteurs dans ChromaDB dans le répertoire `chroma_db/`

### Ajout de nouvelles données

Lorsque vous ajoutez de nouveaux documents :

```bash
# 1. Ajouter des fichiers au répertoire data
cp /path/to/new_documents/*.html /home/rag/data/

# 2. Ré-exécuter l'ingestion
cd /home/rag && python3 ingest_html_adaptive.py

# 3. Redémarrer le service
bash /home/rag/restart_server.sh
```

---

## Gestion des services

### Démarrage du service

**Méthode A : Script de redémarrage (Recommandée)**
```bash
bash /home/rag/restart_server.sh
```

**Méthode B : Démarrage manuel**
```bash
cd /home/rag && nohup python3 server.py > server.log 2>&1 &
```

**Méthode C : Utilisation de systemd (Production)**
```bash
# 1. Copier le fichier de service
sudo cp /home/rag/rag-api.service /etc/systemd/system/

# 2. Recharger systemd
sudo systemctl daemon-reload

# 3. Activer et démarrer
sudo systemctl enable rag-api
sudo systemctl start rag-api
```

### Arrêt du service

```bash
# Méthode A : Tuer le processus
pkill -f "python3 server.py"

# Méthode B : Systemd
sudo systemctl stop rag-api
```

### Vérification du statut du service

```bash
# Vérifier si le processus fonctionne
ps aux | grep "python3 server.py"

# Vérifier les logs
cat /home/rag/server.log

# Suivre les logs en temps réel
tail -f /home/rag/server.log

# Vérifier les logs (méthode systemd)
sudo journalctl -u rag-api -f
```

### Test de l'API

```bash
# Vérification de santé
curl http://localhost:8000/health

# Poser une question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Votre question ici"}'
```

---

## Dépannage

### Problème 1 : Erreurs de permission refusée

**Symptôme :**
```
bash: /home/rag/file.txt: Permission denied
```

**Solution :**
```bash
# Corriger la propriété
sudo chown -R $USER:$USER /home/rag
```

### Problème 2 : Erreur ChromaDB "Nothing found on disk"

**Symptôme :**
```
Error creating hnsw segment reader: Nothing found on disk
```

**Cause :** Service démarré avant l'ingestion des données, ou ingestion échouée.

**Solution :**
```bash
# 1. Arrêter le service
pkill -f "python3 server.py"

# 2. Vérifier que les données existent
ls -la /home/rag/data/

# 3. Ré-exécuter l'ingestion
cd /home/rag && python3 ingest_html_adaptive.py

# 4. Redémarrer le service
bash /home/rag/restart_server.sh
```

### Problème 3 : Réponses génériques (Ne utilisant pas les données RAG)

**Symptôme :** L'API répond mais n'utilise pas les données ingérées.

**Diagnostic :**
```bash
# Vérifier si ChromaDB existe
ls -la /home/rag/chroma_db/

# Vérifier les logs d'ingestion
cat /home/rag/server.log
```

**Solution :** Suivre les étapes du Problème 2.

### Problème 4 : Modèles Ollama introuvables

**Symptôme :**
```
Model 'mistral:7b' not found
```

**Solution :**
```bash
# Lister les modèles disponibles
ollama list

# Tirer les modèles manquants
ollama pull mistral:7b
ollama pull nomic-embed-text
```

### Problème 5 : Port 8000 déjà utilisé

**Symptôme :**
```
ERROR: [Errno 98] Address already in use
```

**Solution :**
```bash
# Trouver le processus utilisant le port 8000
lsof -i :8000

# Tuer le processus
kill -9 <PID>

# Ou utiliser le script de redémarrage qui gère cela automatiquement
bash /home/rag/restart_server.sh
```

### Problème 6 : Dépendances Python manquantes

**Symptôme :**
```
ModuleNotFoundError: No module named 'langchain'
```

**Solution :**
```bash
# Réinstaller les dépendances
pip3 install \
    langchain langchain-community langchain-chroma \
    langchain-ollama chromadb fastapi uvicorn python-dotenv beautifulsoup4
```

---

## Maintenance

### Surveillance

```bash
# Vérifier les logs du serveur
tail -f /home/rag/server.log

# Surveiller l'utilisation des ressources
htop
```

### Sauvegarde

```bash
# Sauvegarder les données et la base de données
tar -czf rag-backup-$(date +%Y%m%d).tar.gz \
    /home/rag/data \
    /home/rag/chroma_db
```

### Mise à jour des modèles

```bash
# Mettre à jour les modèles Ollama
ollama pull mistral:7b
ollama pull nomic-embed-text

# Redémarrer le service pour utiliser les modèles mis à jour
bash /home/rag/restart_server.sh
```

### Effacement et ré-ingestion des données

```bash
# Arrêter le service
pkill -f "python3 server.py"

# Supprimer l'ancienne base de données
rm -rf /home/rag/chroma_db

# Effacer le répertoire data (optionnel)
rm -rf /home/rag/data/*

# Ajouter de nouvelles données
cp /path/to/new/data/*.html /home/rag/data/

# Ré-ingérer
cd /home/rag && python3 ingest_html_adaptive.py

# Redémarrer le service
bash /home/rag/restart_server.sh
```

---

## Référence rapide des commandes

| Action | Commande |
|--------|----------|
| Démarrer le service | `bash /home/rag/restart_server.sh` |
| Arrêter le service | `pkill -f "python3 server.py"` |
| Vérifier le statut | `ps aux \| grep "python3 server.py"` |
| Voir les logs | `cat /home/rag/server.log` |
| Tester l'API | `curl http://localhost:8000/health` |
| Ingérer les données | `cd /home/rag && python3 ingest_html_adaptive.py` |
| Lister les modèles | `ollama list` |
| Changer le modèle | `bash /home/rag/change_model.sh <nom_modele>` |
| Vérifier mode GPU/CPU | `python3 -c "from device_config import print_device_status; print_device_status()"` |
| Activer GPU | `export RAG_USE_GPU=1` |

---

## Résumé de l'architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (curl/navigateur)              │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST /ask
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Serveur FastAPI (server.py)                 │
│                  Port 8000                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Pipeline RAG (rag_pipeline.py)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   ChromaDB   │  │  Intégrations│  │   Modèle LLM │  │
│  │  (Retriever) │  │ nomic-embed  │  │  mistral:7b  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │ Ingestion (ingest_html_adaptive.py)
                         │
┌─────────────────────────────────────────────────────────┐
│              Fichiers de données (data/*.html)            │
└─────────────────────────────────────────────────────────┘
```

---

## Support

Pour les problèmes ou questions :
1. Vérifiez la section [Dépannage](#dépannage)
2. Examinez les logs : `/home/rag/server.log`
3. Vérifiez que tous les prérequis sont satisfaits
4. Assurez-vous que le service Ollama fonctionne : `systemctl status ollama`
