#!/bin/bash
# ============================================================================
# Script d'Ingestion HTML OPTIMISÉ - Avec extraction d'images
# ============================================================================
# Ce script utilise la version améliorée qui extrait les URLs d'images
# Usage: sudo bash ingest_html_optimized_workflow.sh [chemin_source_html]
# ============================================================================

set -e  # Exit on error

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
RAG_HOME="/home/ragapp/rag-system"
DATA_DIR="${RAG_HOME}/data"
VENV_PYTHON="${RAG_HOME}/venv/bin/python3"
VENV_PIP="${RAG_HOME}/venv/bin/pip"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  INGESTION HTML OPTIMISÉE - RAG       ${NC}"
echo -e "${BLUE}  (avec extraction d'images)           ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# ============================================================================
# ÉTAPE 1 : Vérifications préliminaires
# ============================================================================
echo -e "${YELLOW}[1/6] Vérifications préliminaires...${NC}"

# Vérifier les permissions
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}✗ Ce script doit être exécuté avec sudo${NC}"
    exit 1
fi

# Vérifier que le répertoire RAG existe
if [ ! -d "$RAG_HOME" ]; then
    echo -e "${RED}✗ Le répertoire $RAG_HOME n'existe pas${NC}"
    exit 1
fi

# Créer le répertoire data s'il n'existe pas
if [ ! -d "$DATA_DIR" ]; then
    sudo -u ragapp mkdir -p "$DATA_DIR"
    echo -e "${GREEN}✓ Répertoire data créé${NC}"
fi

echo -e "${GREEN}✓ Vérifications OK${NC}"
echo ""

# ============================================================================
# ÉTAPE 2 : Copier les fichiers HTML
# ============================================================================
echo -e "${YELLOW}[2/6] Copie des fichiers HTML...${NC}"

if [ -n "$1" ]; then
    SOURCE_PATH="$1"
    echo "Source: $SOURCE_PATH"
    
    if [ -d "$SOURCE_PATH" ]; then
        # C'est un répertoire
        HTML_COUNT=$(find "$SOURCE_PATH" -name "*.html" -o -name "*.htm" | wc -l)
        if [ "$HTML_COUNT" -eq 0 ]; then
            echo -e "${RED}✗ Aucun fichier HTML trouvé dans $SOURCE_PATH${NC}"
            exit 1
        fi
        
        echo "Copie de $HTML_COUNT fichiers HTML..."
        cp -r "$SOURCE_PATH"/*.html "$DATA_DIR/" 2>/dev/null || \
        cp -r "$SOURCE_PATH"/*.htm "$DATA_DIR/" 2>/dev/null || true
        
    elif [ -f "$SOURCE_PATH" ]; then
        # C'est un fichier unique
        cp "$SOURCE_PATH" "$DATA_DIR/"
        HTML_COUNT=1
    else
        echo -e "${RED}✗ Le chemin $SOURCE_PATH n'existe pas${NC}"
        exit 1
    fi
    
    # Corriger les permissions
    chown -R ragapp:ragapp "$DATA_DIR"
    echo -e "${GREEN}✓ $HTML_COUNT fichier(s) copié(s)${NC}"
else
    echo -e "${YELLOW}ℹ Aucun chemin source fourni${NC}"
    echo "Les fichiers HTML doivent déjà être dans $DATA_DIR"
fi

# Compter les fichiers HTML dans data
TOTAL_HTML=$(find "$DATA_DIR" -name "*.html" -o -name "*.htm" | wc -l)
echo "Fichiers HTML dans data/: $TOTAL_HTML"

if [ "$TOTAL_HTML" -eq 0 ]; then
    echo -e "${RED}✗ Aucun fichier HTML trouvé dans $DATA_DIR${NC}"
    echo "Copiez d'abord vos fichiers HTML dans ce répertoire"
    exit 1
fi

echo ""

# ============================================================================
# ÉTAPE 3 : Installer les dépendances
# ============================================================================
echo -e "${YELLOW}[3/6] Installation des dépendances...${NC}"

# Liste des dépendances à installer
DEPS_INSTALLED=0

# Vérifier unstructured
if ! sudo -u ragapp bash -c "$VENV_PYTHON -c 'import unstructured' 2>/dev/null"; then
    echo "Installation de 'unstructured'..."
    sudo -u ragapp bash -c "cd $RAG_HOME && source venv/bin/activate && pip install -q unstructured[html] && deactivate"
    DEPS_INSTALLED=1
fi

# Vérifier beautifulsoup4
if ! sudo -u ragapp bash -c "$VENV_PYTHON -c 'import bs4' 2>/dev/null"; then
    echo "Installation de 'beautifulsoup4'..."
    sudo -u ragapp bash -c "cd $RAG_HOME && source venv/bin/activate && pip install -q beautifulsoup4 && deactivate"
    DEPS_INSTALLED=1
fi

if [ $DEPS_INSTALLED -eq 0 ]; then
    echo -e "${GREEN}✓ Toutes les dépendances sont déjà installées${NC}"
else
    echo -e "${GREEN}✓ Nouvelles dépendances installées${NC}"
fi

echo ""

# ============================================================================
# ÉTAPE 4 : Exécuter l'ingestion optimisée
# ============================================================================
echo -e "${YELLOW}[4/6] Ingestion optimisée des documents HTML...${NC}"
echo "📝 Extraction du texte + URLs d'images en cours..."
echo "⏱ Cela peut prendre du temps selon le nombre de fichiers..."
echo ""

# Exécuter le script d'ingestion optimisé
sudo -u ragapp bash -c "cd $RAG_HOME && $VENV_PYTHON ingest_html_optimized.py"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ingestion réussie${NC}"
else
    echo -e "${RED}✗ Erreur lors de l'ingestion${NC}"
    exit 1
fi

echo ""

# ============================================================================
# ÉTAPE 5 : Redémarrer le serveur
# ============================================================================
echo -e "${YELLOW}[5/6] Redémarrage du serveur RAG...${NC}"

# Arrêter le serveur existant
if pgrep -f "uvicorn server:app" > /dev/null; then
    pkill -f "uvicorn server:app"
    echo "Serveur arrêté"
    sleep 2
fi

# Redémarrer le serveur
sudo -u ragapp bash -c "cd $RAG_HOME && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"

# Attendre le démarrage
echo "Attente du démarrage du serveur..."
sleep 5

if pgrep -f "uvicorn server:app" > /dev/null; then
    echo -e "${GREEN}✓ Serveur redémarré${NC}"
else
    echo -e "${RED}✗ Erreur au démarrage du serveur${NC}"
    echo "Vérifiez les logs: cat $RAG_HOME/server.log"
    exit 1
fi

echo ""

# ============================================================================
# ÉTAPE 6 : Vérification
# ============================================================================
echo -e "${YELLOW}[6/6] Vérification du système...${NC}"

# Test de santé
HEALTH_CHECK=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_CHECK" | grep -q "ok"; then
    echo -e "${GREEN}✓ Health check OK${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi

# Vérifier ChromaDB
if [ -d "$RAG_HOME/chroma_db" ]; then
    DB_SIZE=$(du -sh "$RAG_HOME/chroma_db" | cut -f1)
    echo -e "${GREEN}✓ Base ChromaDB créée (taille: $DB_SIZE)${NC}"
fi

echo ""

# ============================================================================
# RÉSUMÉ
# ============================================================================
echo -e "${BLUE}==========================================${NC}"
echo -e "${GREEN}   INGESTION OPTIMISÉE TERMINÉE !      ${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""
echo -e "${GREEN}📊 Résumé:${NC}"
echo "  - Fichiers HTML traités: $TOTAL_HTML"
echo "  - Serveur: http://localhost:8000"
echo "  - Base de données: $RAG_HOME/chroma_db"
echo ""
echo -e "${GREEN}✨ Améliorations appliquées:${NC}"
echo "  ✓ URLs d'images extraites et indexées"
echo "  ✓ Séparateurs optimisés (listes, titres)"
echo "  ✓ Chunks adaptés aux procédures (800 chars)"
echo "  ✓ Overlap optimisé (100 chars = 12.5%)"
echo ""
echo -e "${BLUE}🧪 Test rapide:${NC}"
echo "curl -X POST \"http://localhost:8000/ask\" \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"question\": \"Où se trouvent les images ?\"}'"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo "  - Serveur: $RAG_HOME/server.log"
echo ""
