# config_searchnl.py
"""
Configuration pour SearchNL - Transformation du langage naturel (NL) vers XFT.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "config_searchnl.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ajoute le répertoire parent au chemin pour l'importation de la configuration
logger.debug("Ajout du répertoire parent au sys.path")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ACTIVE_MODEL, MODELS

# =============================================================================
# CONFIGURATION LLM
# =============================================================================

# Récupère la configuration du modèle à partir de la configuration principale
logger.debug(f"Modèle actif: {ACTIVE_MODEL}")
MODEL_CONFIG = MODELS.get(ACTIVE_MODEL, MODELS["qwen-14b"])
MODEL_NAME = MODEL_CONFIG["name"]
logger.info(f"Configuration LLM chargée: {MODEL_NAME}")

# Paramètres Ollama pour l'extraction d'entités
OLLAMA_CONFIG = {
    "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "timeout": 30,
    "temperature": 0.1,  # Basse température pour un parsing cohérent
    "num_predict": 256,  # Réponses courtes nécessaires
}
logger.debug(f"Configuration Ollama: {OLLAMA_CONFIG}")

# =============================================================================
# INVITES DU PARSEUR NL
# =============================================================================

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'extraction d'informations de vol à partir de requêtes en langage naturel.
Extrais les informations suivantes et retourne-les au format JSON strict:

- origin: ville ou aéroport de départ
- destination: ville ou aéroport d'arrivée  
- date: date de départ (format YYYY-MM-DD si possible, sinon la date mentionnée)
- return_date: date de retour si mentionnée (null sinon)
- adults: nombre d'adultes (défaut: 1)
- children: nombre d'enfants 2-11 ans (défaut: 0)
- infants: nombre de bébés <2 ans (défaut: 0)
- trip_type: "OW" (aller simple), "RT" (aller-retour), ou "MC" (multi-destinations)
- cabin_class: "economy", "premium_economy", "business", "first" (défaut: "economy")
- flexible_dates: true/false si les dates sont flexibles
- direct_only: true/false si vol direct uniquement demandé

Réponds UNIQUEMENT avec le JSON, sans explication."""

EXTRACTION_PROMPT_TEMPLATE = """Requête utilisateur: {query}

Date actuelle: {current_date}

Extrais les informations de vol au format JSON."""

# =============================================================================
# CONFIGURATION DE L'API
# =============================================================================

API_CONFIG = {
    "host": os.getenv("SEARCHNL_HOST", "0.0.0.0"),
    "port": int(os.getenv("SEARCHNL_PORT", "8001")),
    "debug": os.getenv("SEARCHNL_DEBUG", "false").lower() == "true",
}
logger.debug(f"Configuration API: {API_CONFIG}")

# =============================================================================
# CONFIGURATION XFT
# =============================================================================

XFT_CONFIG = {
    "templates_dir": os.path.join(os.path.dirname(__file__), "XFT"),
    "default_code_type": "metropolitan",  # "airport" (aéroport) ou "metropolitan" (métropolitain)
    "validate_xml": True,
}
logger.debug(f"Configuration XFT: {XFT_CONFIG}")

# =============================================================================
# CONFIGURATION DE LA BASE DE DONNÉES DES AÉROPORTS
# =============================================================================

AIRPORT_CONFIG = {
    "fuzzy_threshold": 80,  # Score de correspondance floue minimum (0-100)
    "prefer_metropolitan": True,  # Utiliser les codes de ville lorsque disponibles
}
logger.debug(f"Configuration Airport: {AIRPORT_CONFIG}")

logger.info("Configuration SearchNL chargée avec succès")


if __name__ == "__main__":
    logger.info("Affichage de la configuration SearchNL")
    print(f"Configuration SearchNL")
    print(f"  Modèle: {MODEL_NAME}")
    print(f"  Port API: {API_CONFIG['port']}")
    print(f"  Modèles XFT: {XFT_CONFIG['templates_dir']}")
