# Guide de Déploiement Entreprise - RAG System

Ce guide couvre l'installation complète d'un système RAG sur un serveur **Ubuntu Linux** en mode production avec API REST.

---

## 📋 Prérequis

### Serveur
- **OS**: Ubuntu 22.04 LTS ou supérieur
- **RAM**: Minimum 8 GB (16 GB recommandé)
- **Stockage**: 20 GB d'espace libre
- **Accès**: SSH avec droits sudo

### GPU (Optionnel mais recommandé)
- NVIDIA GPU avec 8GB+ VRAM
- Drivers NVIDIA installés

---

## 🚀 Installation Complète

### Étape 1: Connexion et Mise à Jour du Système

```bash
# Connexion SSH
ssh user@your-server-ip

# Mise à jour du système
sudo apt update && sudo apt upgrade -y
```

### Étape 2: Installation de Python et Dépendances

```bash
# Installation de Python 3.11+
sudo apt install -y python3 python3-pip python3-venv git curl

# Vérification
python3 --version  # Doit être >= 3.10
```

### Étape 3: Installation d'Ollama

```bash
# Téléchargement et installation d'Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Vérification
ollama --version
```

### Étape 4: Configuration GPU vs CPU

#### Option A: Configuration GPU (Recommandé si disponible)

```bash
# Vérifier la présence du GPU
nvidia-smi

# Si le GPU est détecté, Ollama l'utilisera automatiquement
# Aucune configuration supplémentaire nécessaire
```

#### Option B: Configuration CPU-Only

Si vous n'avez pas de GPU ou si vous voulez forcer l'utilisation du CPU:

```bash
# Créer un fichier de configuration Ollama
sudo mkdir -p /etc/systemd/system/ollama.service.d/
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

Ajouter le contenu suivant:
```ini
[Service]
Environment="OLLAMA_NUM_GPU=0"
```

Puis recharger:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Étape 5: Téléchargement des Modèles

```bash
# Modèle LLM (Llama 3.2 - 2GB)
ollama pull mistral:7b

# Modèle d'embeddings (274 MB)
ollama pull nomic-embed-text

# Vérification
ollama list
```

### Étape 6: Création de l'Utilisateur et du Répertoire

```bash
# Créer un utilisateur dédié (bonne pratique de sécurité)
sudo useradd -m -s /bin/bash ragapp
sudo usermod -aG sudo ragapp

# Se connecter en tant que ragapp
sudo su - ragapp

# Créer le répertoire de l'application
mkdir -p ~/rag-system
cd ~/rag-system
```

### Étape 7: Déploiement du Code

```bash
# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Créer requirements.txt
cat > requirements.txt << 'EOF'
langchain
langchain-openai
langchain-community
chromadb
langchain-chroma
langchain-ollama
python-dotenv
fastapi
uvicorn
EOF

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 8: Copie des Fichiers de Code

Transférer les fichiers depuis votre PC Windows vers le serveur:

```bash
# Depuis votre PC Windows (PowerShell)
scp -r c:/coding/rag/* ragapp@your-server-ip:~/rag-system/

# Ou utiliser WinSCP / FileZilla pour transférer:
# - rag_pipeline.py
# - server.py
# - data/sample.txt
# - requirements.txt
```

### Étape 9: Configuration du Service Systemd

Créer un service pour que le RAG démarre automatiquement:

```bash
# Revenir en tant que root/sudo
exit  # Quitter l'utilisateur ragapp

# Créer le fichier de service
sudo nano /etc/systemd/system/rag-api.service
```

Contenu du fichier:
```ini
[Unit]
Description=RAG Enterprise API Service
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ragapp
WorkingDirectory=/home/ragapp/rag-system
Environment="PATH=/home/ragapp/rag-system/venv/bin"
ExecStart=/home/ragapp/rag-system/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activer et démarrer le service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl start rag-api

# Vérifier le statut
sudo systemctl status rag-api
```

### Étape 10: Configuration du Firewall

```bash
# Autoriser le port 8000
sudo ufw allow 8000/tcp

# Vérifier
sudo ufw status
```

---

## 🧪 Tests et Vérification

### Test Local (sur le serveur)

```bash
# Test de santé
curl http://localhost:8000/health

# Test de question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la hauteur de la Tour Eiffel ?"}'
```

### Test Distant (depuis votre PC)

```bash
# Remplacer YOUR_SERVER_IP par l'IP réelle
curl -X POST http://YOUR_SERVER_IP:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qui a créé Python ?"}'
```

---

## 📊 Monitoring et Logs

### Voir les logs en temps réel

```bash
# Logs du service RAG
sudo journalctl -u rag-api -f

# Logs d'Ollama
sudo journalctl -u ollama -f
```

### Redémarrer le service

```bash
sudo systemctl restart rag-api
```

---

## 🔒 Sécurité (Production)

### 1. Reverse Proxy avec Nginx

```bash
sudo apt install -y nginx

sudo nano /etc/nginx/sites-available/rag-api
```

Contenu:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Activer:
```bash
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 2. HTTPS avec Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Authentification API (Optionnel)

Ajouter une clé API dans `server.py`:
```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key"

@app.post("/ask")
async def ask_question(request: QueryRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    # ... reste du code
```

---

## 🔧 Optimisations

### Pour GPU (si disponible)

Modifier `rag_pipeline.py`:
```python
# Retirer num_gpu=0 dans les configurations OllamaEmbeddings et ChatOllama
# Le GPU sera utilisé automatiquement
```

### Pour CPU (performances)

```bash
# Augmenter le nombre de threads CPU pour Ollama
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

Ajouter:
```ini
[Service]
Environment="OLLAMA_NUM_THREAD=8"
```

---

## 📁 Structure Finale du Serveur

```
/home/ragapp/rag-system/
├── venv/                    # Environnement virtuel Python
├── chroma_db/               # Base de données vectorielle
├── data/
│   └── sample.txt          # Données sources
├── rag_pipeline.py         # Pipeline RAG
├── server.py               # API FastAPI
├── main.py                 # Script CLI (optionnel)
├── requirements.txt        # Dépendances Python
└── .env                    # Variables d'environnement (si nécessaire)
```

---

## ❓ Troubleshooting

### Le service ne démarre pas
```bash
# Vérifier les logs
sudo journalctl -u rag-api -n 50

# Vérifier qu'Ollama fonctionne
sudo systemctl status ollama
ollama list
```

### Erreur "Out of Memory"
- Réduire `num_ctx` dans `rag_pipeline.py` (ex: 1024 au lieu de 2048)
- Forcer CPU-only si le GPU manque de VRAM

### Performances lentes
- Vérifier que le GPU est utilisé: `nvidia-smi` (si disponible)
- Augmenter la RAM du serveur
- Utiliser un modèle plus petit (ex: `llama3.2:1b`)

---

## 🎯 Prochaines Étapes

1. **Ajouter vos propres données** dans `data/` (PDF, TXT, etc.)
2. **Adapter le chunking** dans `rag_pipeline.py` selon vos besoins
3. **Configurer un load balancer** si vous avez plusieurs serveurs
4. **Mettre en place des métriques** (Prometheus, Grafana)

---

**Auteur**: Guide généré pour déploiement entreprise  
**Version**: 1.0  
**Date**: 2025-11-26
