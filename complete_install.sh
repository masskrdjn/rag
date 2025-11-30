#!/bin/bash
# ============================================================================
# SYSTÈME RAG - PROCÉDURE D'INSTALLATION COMPLÈTE
# ============================================================================
# Ce script contient toutes les commandes nécessaires pour installer et configurer
# le système RAG à partir de zéro sur Ubuntu 22.04 (testé sur WSL2).
#
# Prérequis :
# - Ubuntu 22.04 ou ultérieur
# - Privilèges sudo
# - Connexion Internet
#
# Utilisation : sudo bash complete_install.sh
# ============================================================================

set -e  # Quitter en cas d'erreur

echo "=========================================="
echo "SYSTÈME RAG - INSTALLATION COMPLÈTE"
echo "=========================================="
echo ""

# ============================================================================
# PARTIE 1 : DÉPENDANCES SYSTÈME ET CONFIGURATION UTILISATEUR
# ============================================================================
echo "[1/7] Installation des dépendances système..."

# Mettre à jour le système
apt update && apt upgrade -y

# Installer Python et ses dépendances
apt install -y python3 python3-pip python3-venv git curl

# Installer Ollama
echo "Installation d'Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Créer un utilisateur dédié pour l'application RAG
echo "Création de l'utilisateur ragapp..."
if ! id -u ragapp > /dev/null 2>&1; then
    useradd -m -s /bin/bash ragapp
    echo "Utilisateur 'ragapp' créé avec succès"
else
    echo "L'utilisateur 'ragapp' existe déjà, ignorer..."
fi

# Assurer la propriété du répertoire personnel
chown ragapp:ragapp /home/ragapp

echo "[1/7] Dépendances système installées ✓"
echo ""

# ============================================================================
# PARTIE 2 : TÉLÉCHARGEMENT DES MODÈLES D'IA
# ============================================================================
echo "[2/7] Téléchargement des modèles d'IA (cela peut prendre plusieurs minutes)..."

# Télécharger les modèles Ollama requis
sudo -u ragapp ollama pull mistral:7b
sudo -u ragapp ollama pull nomic-embed-text

echo "[2/7] Modèles d'IA téléchargés ✓"
echo ""

# ============================================================================
# PARTIE 3 : CONFIGURATION DE L'APPLICATION
# ============================================================================
echo "[3/7] Configuration du répertoire de l'application..."

# Créer le répertoire de l'application
sudo -u ragapp mkdir -p /home/ragapp/rag-system
sudo -u ragapp mkdir -p /home/ragapp/rag-system/data

# Note : À ce stade, copiez vos fichiers d'application dans /home/ragapp/rag-system/
echo "ÉTAPE MANUELLE : Copiez les fichiers suivants dans /home/ragapp/rag-system/ :"
echo "  - server.py"
echo "  - rag_pipeline.py"
echo "  - ingest_adaptive.py"
echo "  - rag-api.service (pour systemd, facultatif)"
echo ""
read -p "Appuyez sur Entrée une fois les fichiers copiés..."

# Corriger la propriété
chown -R ragapp:ragapp /home/ragapp/rag-system

echo "[3/7] Répertoire de l'application prêt ✓"
echo ""

# ============================================================================
# PARTIE 4 : ENVIRONNEMENT VIRTUEL PYTHON
# ============================================================================
echo "[4/7] Création de l'environnement virtuel Python..."

cd /home/ragapp/rag-system

# Créer l'environnement virtuel en tant qu'utilisateur ragapp
sudo -u ragapp python3 -m venv venv

# Mettre à jour pip
sudo -u ragapp /home/ragapp/rag-system/venv/bin/pip install --upgrade pip

# Installer les paquets Python requis
sudo -u ragapp /home/ragapp/rag-system/venv/bin/pip install \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-ollama \
    chromadb \
    fastapi \
    uvicorn \
    python-dotenv

echo "[4/7] Environnement Python configuré ✓"
echo ""

# ============================================================================
# PARTIE 5 : INGESTION DES DONNÉES
# ============================================================================
echo "[5/7] Ingestion des données dans ChromaDB..."

# Créer des données d'exemple si besoin
# Note: Pour un usage en production, placez vos fichiers HTML dans /home/ragapp/rag-system/data/
# Exemple de création d'une structure de test :
# sudo -u ragapp bash -c "echo '<html><body><h1>Test</h1><p>Contenu de test</p></body></html>' > /home/ragapp/rag-system/data/test.html"

# Exécuter le script d'ingestion
echo "Exécution du script d'ingestion de données..."
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest_adaptive.py"

echo "[5/7] Ingestion des données terminée ✓"
echo ""

# ============================================================================
# PARTIE 6 : CONFIGURATION DU SERVICE (FACULTATIF - Choisissez une méthode)
# ============================================================================
echo "[6/7] Démarrage du service..."

# Méthode A : Utilisation de systemd (recommandé pour la production)
# Décommentez les lignes suivantes si vous avez configuré rag-api.service
# cp /home/ragapp/rag-system/rag-api.service /etc/systemd/system/
# systemctl daemon-reload
# systemctl enable rag-api
# systemctl start rag-api

# Méthode B : Utilisation de nohup (plus simple, pour le développement/test)
echo "Démarrage du serveur avec nohup..."
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"

# Attendre le démarrage du serveur
sleep 5

echo "[6/7] Service démarré ✓"
echo ""

# ============================================================================
# PARTIE 7 : VÉRIFICATION
# ============================================================================
echo "[7/7] Vérification de l'installation..."

# Vérifier si le processus est en cours d'exécution
if pgrep -f "uvicorn server:app" > /dev/null; then
    echo "✓ Le processus du serveur est en cours d'exécution"
else
    echo "✗ Processus du serveur introuvable - vérifiez les journaux"
fi

# Tester l'API
echo "Test du point de terminaison de l'API..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Quelle est la hauteur de la Tour Eiffel ?"}')

if echo "$RESPONSE" | grep -q "330"; then
    echo "✓ Test de l'API réussi !"
    echo "Réponse : $RESPONSE"
else
    echo "✗ Test de l'API échoué - réponse inattendue"
    echo "Réponse : $RESPONSE"
fi

echo "[7/7] Vérification terminée ✓"
echo ""

# ============================================================================
# INSTALLATION TERMINÉE
# ============================================================================
echo "=========================================="
echo "INSTALLATION TERMINÉE !"
echo "=========================================="
echo ""
echo "Informations sur le service :"
echo "  - URL : http://localhost:8000"
echo "  - Vérification de l'état : http://localhost:8000/health"
echo "  - Point de terminaison de l'API : POST http://localhost:8000/ask"
echo "  - Journaux : /home/ragapp/rag-system/server.log"
echo ""
echo "Exemple d'utilisation :"
echo '  curl -X POST "http://localhost:8000/ask" \'
echo '       -H "Content-Type: application/json" \'
echo "       -d '{\"question\": \"Votre question ici\"}'"
echo ""
echo "Pour ajouter plus de données :"
echo "  1. Placez les fichiers HTML dans /home/ragapp/rag-system/data/"
echo "  2. Exécutez : sudo -u ragapp bash -c 'cd /home/ragapp/rag-system && venv/bin/python3 ingest_adaptive.py'"
echo "  3. Redémarrez le service : sudo pkill -9 -f uvicorn && sudo -u ragapp bash -c 'cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &'"
echo ""
echo "Dépannage :"
echo "  - Vérifiez les journaux : sudo cat /home/ragapp/rag-system/server.log"
echo "  - Vérifiez si Ollama est en cours d'exécution : systemctl status ollama"
echo "  - Vérifiez les modèles : sudo -u ragapp ollama list"
echo "  - Tuez les processus bloqués : sudo pkill -f uvicorn"
echo ""
