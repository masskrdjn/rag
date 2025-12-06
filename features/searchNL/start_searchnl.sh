#!/bin/bash
# start_searchnl.sh
# Démarre le serveur API SearchNL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "🔍 SearchNL - Natural Language to XFT API"
echo "=========================================="

# Vérifie si l'environnement virtuel existe
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Environnement virtuel activé"
fi

# Vérifie si Ollama est en cours d'exécution
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠ Avertissement : Ollama ne semble pas être en cours d'exécution sur le port 11434"
    echo "  Démarrez Ollama avec : ollama serve"
fi

# Définit les variables d'environnement
export SEARCHNL_PORT=${SEARCHNL_PORT:-8001}
export SEARCHNL_HOST=${SEARCHNL_HOST:-0.0.0.0}

echo ""
echo "Démarrage du serveur sur http://$SEARCHNL_HOST:$SEARCHNL_PORT"
echo "Documentation API : http://localhost:$SEARCHNL_PORT/docs"
echo ""

# Exécute l'API
cd "$PROJECT_ROOT"
python -m features.searchNL.api_handler
