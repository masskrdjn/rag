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
# Répertoire d'installation : /home/rag/
#
# Utilisation : sudo bash complete_install.sh
# ============================================================================

set -e  # Quitter en cas d'erreur

# Répertoire d'installation
RAG_DIR="/home/rag"

echo "=========================================="
echo "SYSTÈME RAG - INSTALLATION COMPLÈTE"
echo "=========================================="
echo "Répertoire d'installation : $RAG_DIR"
echo ""

# ============================================================================
# PARTIE 1 : DÉPENDANCES SYSTÈME
# ============================================================================
echo "[1/6] Installation des dépendances système..."

# Mettre à jour le système
apt update && apt upgrade -y

# Installer Python et ses dépendances
apt install -y python3 python3-pip python3-venv git curl

# Installer Ollama
echo "Installation d'Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "[1/6] Dépendances système installées ✓"
echo ""

# ============================================================================
# PARTIE 2 : TÉLÉCHARGEMENT DES MODÈLES D'IA
# ============================================================================
echo "[2/6] Téléchargement des modèles d'IA (cela peut prendre plusieurs minutes)..."

# Télécharger les modèles Ollama requis
ollama pull mistral:7b
ollama pull nomic-embed-text

echo "[2/6] Modèles d'IA téléchargés ✓"
echo ""

# ============================================================================
# PARTIE 3 : CONFIGURATION DE L'APPLICATION
# ============================================================================
echo "[3/6] Configuration du répertoire de l'application..."

# Créer le répertoire de l'application
mkdir -p $RAG_DIR
mkdir -p $RAG_DIR/data

# Corriger la propriété
chown -R $SUDO_USER:$SUDO_USER $RAG_DIR

# Vérifier que les fichiers principaux sont présents
if [ ! -f "$RAG_DIR/server.py" ] || [ ! -f "$RAG_DIR/rag_pipeline.py" ] || [ ! -f "$RAG_DIR/ingest_html_adaptive.py" ]; then
    echo "ATTENTION : Fichiers manquants dans $RAG_DIR/"
    echo "Fichiers requis :"
    echo "  - server.py"
    echo "  - rag_pipeline.py"
    echo "  - ingest_html_adaptive.py"
    echo ""
    read -p "Appuyez sur Entrée une fois les fichiers copiés..."
fi

echo "[3/6] Répertoire de l'application prêt ✓"
echo ""

# ============================================================================
# PARTIE 4 : DÉPENDANCES PYTHON
# ============================================================================
echo "[4/6] Installation des dépendances Python..."

# Mettre à jour pip
pip3 install --upgrade pip

# Installer les paquets Python requis
pip3 install \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-ollama \
    chromadb \
    fastapi \
    uvicorn \
    python-dotenv \
    beautifulsoup4

echo "[4/6] Dépendances Python installées ✓"
echo ""

# ============================================================================
# PARTIE 5 : INGESTION DES DONNÉES
# ============================================================================
echo "[5/6] Ingestion des données dans ChromaDB..."

# Vérifier s'il y a des fichiers HTML dans data/
if [ -z "$(ls -A $RAG_DIR/data/*.html 2>/dev/null)" ]; then
    echo "ATTENTION : Aucun fichier HTML trouvé dans $RAG_DIR/data/"
    echo "Placez vos fichiers HTML dans ce répertoire avant l'ingestion."
else
    # Exécuter le script d'ingestion
    echo "Exécution du script d'ingestion de données..."
    cd $RAG_DIR && python3 ingest_html_adaptive.py
fi

echo "[5/6] Ingestion des données terminée ✓"
echo ""

# ============================================================================
# PARTIE 6 : DÉMARRAGE DU SERVICE
# ============================================================================
echo "[6/6] Démarrage du service..."

# Utilisation du script de redémarrage s'il existe
if [ -f "$RAG_DIR/restart_server.sh" ]; then
    bash $RAG_DIR/restart_server.sh
else
    # Démarrage manuel avec nohup
    cd $RAG_DIR && nohup python3 server.py > server.log 2>&1 &
fi

# Attendre le démarrage du serveur
sleep 5

# Vérifier si le processus est en cours d'exécution
if pgrep -f "python3 server.py" > /dev/null; then
    echo "✓ Le processus du serveur est en cours d'exécution"
else
    echo "✗ Processus du serveur introuvable - vérifiez les journaux"
fi

# Tester l'API
echo "Test du point de terminaison de l'API..."
RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "Erreur de connexion")

if echo "$RESPONSE" | grep -qi "ok\|healthy\|status"; then
    echo "✓ API accessible !"
else
    echo "⚠ API peut ne pas être encore prête. Vérifiez les logs."
fi

echo "[6/6] Vérification terminée ✓"
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
echo "  - Journaux : $RAG_DIR/server.log"
echo ""
echo "Exemple d'utilisation :"
echo '  curl -X POST "http://localhost:8000/ask" \'
echo '       -H "Content-Type: application/json" \'
echo "       -d '{\"question\": \"Votre question ici\"}'"
echo ""
echo "Pour ajouter plus de données :"
echo "  1. Placez les fichiers HTML dans $RAG_DIR/data/"
echo "  2. Exécutez : cd $RAG_DIR && python3 ingest_html_adaptive.py"
echo "  3. Redémarrez le service : bash $RAG_DIR/restart_server.sh"
echo ""
echo "Dépannage :"
echo "  - Vérifiez les journaux : cat $RAG_DIR/server.log"
echo "  - Vérifiez si Ollama est en cours d'exécution : systemctl status ollama"
echo "  - Vérifiez les modèles : ollama list"
echo "  - Tuez les processus bloqués : pkill -f 'python3 server.py'"
echo ""
