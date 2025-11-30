#!/bin/bash
#
# Script de changement de modèle LLM pour le RAG
# Usage: ./change_model.sh <nom_modele>
# Exemple: ./change_model.sh mistral:7b
#

set -e

# Configuration
RAG_DIR="/home/rag"
PIPELINE_FILE="$RAG_DIR/rag_pipeline.py"
SERVER_LOG="$RAG_DIR/server.log"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERREUR]${NC} $1"
}

# Afficher l'usage
usage() {
    echo "Usage: $0 <nom_modele>"
    echo ""
    echo "Exemples:"
    echo "  $0 mistral:7b"
    echo "  $0 llama3.2"
    echo "  $0 mixtral:8x7b"
    echo "  $0 phi3:medium"
    echo ""
    echo "Options:"
    echo "  --list        Afficher les modèles installés"
    echo "  --current     Afficher le modèle actuel"
    echo "  --help, -h    Afficher cette aide"
    exit 1
}

# Afficher les modèles installés
list_models() {
    log_info "Modèles Ollama installés :"
    ollama list
    echo ""
    log_info "Modèle actuellement configuré :"
    get_current_model
}

# Obtenir le modèle actuel
get_current_model() {
    current=$(grep -o 'self.model_name = "[^"]*"' "$PIPELINE_FILE" | head -1 | cut -d'"' -f2)
    echo "$current"
}

# Vérifier que Ollama est disponible
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama n'est pas installé ou pas dans le PATH"
        exit 1
    fi
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_warn "Le service Ollama ne semble pas actif, tentative de démarrage..."
        sudo systemctl start ollama 2>/dev/null || ollama serve &
        sleep 3
    fi
}

# Télécharger le modèle si nécessaire
download_model() {
    local model=$1
    
    log_info "Vérification du modèle $model..."
    
    # Vérifier si le modèle est déjà installé
    if ollama list | grep -q "^$model"; then
        log_success "Le modèle $model est déjà installé"
        return 0
    fi
    
    log_info "Téléchargement du modèle $model (cela peut prendre plusieurs minutes)..."
    if ollama pull "$model"; then
        log_success "Modèle $model téléchargé avec succès"
    else
        log_error "Échec du téléchargement du modèle $model"
        exit 1
    fi
}

# Modifier la configuration
update_config() {
    local model=$1
    local current_model
    
    current_model=$(get_current_model)
    
    log_info "Modification de la configuration..."
    log_info "  Ancien modèle : $current_model"
    log_info "  Nouveau modèle : $model"
    
    # Backup du fichier
    cp "$PIPELINE_FILE" "$PIPELINE_FILE.backup"
    
    # Remplacer le modèle
    sed -i "s/self.model_name = \"[^\"]*\"/self.model_name = \"$model\"/" "$PIPELINE_FILE"
    
    # Vérifier la modification
    new_model=$(get_current_model)
    if [ "$new_model" == "$model" ]; then
        log_success "Configuration mise à jour"
    else
        log_error "Échec de la mise à jour de la configuration"
        log_info "Restauration du backup..."
        cp "$PIPELINE_FILE.backup" "$PIPELINE_FILE"
        exit 1
    fi
}

# Redémarrer le serveur
restart_server() {
    log_info "Redémarrage du serveur RAG..."
    
    # Arrêter le serveur existant
    pkill -f 'python3 server.py' 2>/dev/null || true
    pkill -f 'uvicorn server:app' 2>/dev/null || true
    sleep 3
    
    # Démarrer le serveur
    cd "$RAG_DIR"
    nohup python3 server.py > "$SERVER_LOG" 2>&1 &
    
    log_info "Attente du démarrage (10 secondes)..."
    sleep 10
    
    # Vérifier que le serveur est démarré
    if ps aux | grep -q '[p]ython3 server.py'; then
        log_success "Serveur démarré"
    else
        log_error "Le serveur n'a pas démarré correctement"
        log_info "Consultez les logs : tail -50 $SERVER_LOG"
        exit 1
    fi
}

# Tester le serveur
test_server() {
    log_info "Test du serveur..."
    
    # Test de santé
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_success "Health check OK"
    else
        log_warn "Health check échoué, le serveur peut encore être en cours d'initialisation"
    fi
    
    # Test fonctionnel rapide
    log_info "Test fonctionnel (première requête, peut être lente)..."
    response=$(curl -s -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "Test de connexion"}' \
        --max-time 120)
    
    if [ -n "$response" ] && echo "$response" | grep -q "answer"; then
        log_success "Test fonctionnel OK"
        echo ""
        log_info "Exemple de réponse :"
        echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('answer','')[:200])" 2>/dev/null || echo "$response"
    else
        log_warn "Réponse inattendue du serveur"
        echo "$response"
    fi
}

# === MAIN ===

# Gérer les options
case "${1:-}" in
    --list)
        list_models
        exit 0
        ;;
    --current)
        echo "Modèle actuel : $(get_current_model)"
        exit 0
        ;;
    --help|-h|"")
        usage
        ;;
esac

MODEL="$1"

echo "=============================================="
echo "  Changement de modèle LLM pour le RAG"
echo "=============================================="
echo ""
echo "Nouveau modèle : $MODEL"
echo ""

# Confirmation
read -p "Continuer ? (o/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    log_info "Opération annulée"
    exit 0
fi

echo ""

# Étapes
log_info "=== Étape 1/5 : Vérification d'Ollama ==="
check_ollama
echo ""

log_info "=== Étape 2/5 : Téléchargement du modèle ==="
download_model "$MODEL"
echo ""

log_info "=== Étape 3/5 : Mise à jour de la configuration ==="
update_config "$MODEL"
echo ""

log_info "=== Étape 4/5 : Redémarrage du serveur ==="
restart_server
echo ""

log_info "=== Étape 5/5 : Tests ==="
test_server
echo ""

echo "=============================================="
log_success "Changement de modèle terminé avec succès !"
echo "=============================================="
echo ""
echo "Le RAG utilise maintenant le modèle : $MODEL"
echo ""
echo "Commandes utiles :"
echo "  - Voir les logs    : tail -f $SERVER_LOG"
echo "  - Tester l'API     : curl http://localhost:8000/health"
echo "  - Rollback         : $0 $(get_current_model 2>/dev/null || echo 'llama3.2')"
echo ""
