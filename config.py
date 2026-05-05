# config.py
"""
Configuration centrale pour le système RAG.

Toutes les valeurs sont surchargeables via variables d'environnement :

- RAG_ACTIVE_MODEL    : clé du modèle dans MODELS (ex: "qwen-14b")
- RAG_MODEL           : nom Ollama brut (ex: "mistral:7b") — prioritaire sur RAG_ACTIVE_MODEL
- RAG_EMBEDDING_MODEL : modèle d'embeddings Ollama
- RAG_CHROMA_DB_PATH  : dossier ChromaDB persistant
- RAG_DATA_PATH       : dossier source pour l'ingestion
- RAG_TOP_K           : top_k retrieval
- RAG_MAX_CONTEXT     : taille max du contexte (chars) injecté au LLM
- RAG_USE_HYBRID      : "1"/"0" pour activer/désactiver BM25 hybride
"""

import os
from pathlib import Path

# =============================================================================
# CONFIGURATION DES MODÈLES
# =============================================================================

MODELS = {
    "mistral-7b": {
        "name": "mistral:7b",
        "description": "Mistral 7B - Rapide mais tendance à halluciner",
        "ram_required": "6-8GB",
        "temperature": 0.1,
        "num_predict": 512,
    },
    "qwen-14b": {
        "name": "qwen2.5:14b",
        "description": "Qwen 2.5 14B - Meilleur suivi d'instructions, excellent français",
        "ram_required": "10-12GB",
        "temperature": 0.1,
        "num_predict": 512,
    },
    "qwen-7b": {
        "name": "qwen2.5:7b",
        "description": "Qwen 2.5 7B - Bon compromis, meilleur que mistral",
        "ram_required": "6-8GB",
        "temperature": 0.1,
        "num_predict": 512,
    },
    "command-r-35b": {
        "name": "command-r:35b",
        "description": "Command-R 35B - Optimisé pour RAG, citations précises",
        "ram_required": "20-24GB",
        "temperature": 0.1,
        "num_predict": 512,
    },
    "llama-8b": {
        "name": "llama3.1:8b",
        "description": "Llama 3.1 8B - Rapide et fiable",
        "ram_required": "6-8GB",
        "temperature": 0.1,
        "num_predict": 512,
    },
}

DEFAULT_ACTIVE_MODEL = "qwen-14b"
ACTIVE_MODEL = os.environ.get("RAG_ACTIVE_MODEL", DEFAULT_ACTIVE_MODEL)

# =============================================================================
# CHEMINS (avec défauts adaptatifs Windows / Linux)
# =============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent

def _default_chroma_path() -> str:
    """
    Défaut intelligent :
    - Linux production : /home/rag/chroma_db si le dossier parent existe
    - Sinon (Windows, dev local) : <projet>/chroma_db
    """
    linux_default = Path("/home/rag/chroma_db")
    if linux_default.parent.exists():
        return str(linux_default)
    return str(_PROJECT_ROOT / "chroma_db")

CHROMA_DB_PATH = os.environ.get("RAG_CHROMA_DB_PATH", _default_chroma_path())
DATA_PATH = os.environ.get("RAG_DATA_PATH", str(_PROJECT_ROOT / "data"))

# =============================================================================
# CONFIGURATION RAG
# =============================================================================

def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

RAG_CONFIG = {
    "retrieval_mode": "similarity",
    "top_k": int(os.environ.get("RAG_TOP_K", "6")),
    "score_threshold": 1.5,
    "use_hybrid": _env_bool("RAG_USE_HYBRID", True),
    "hybrid_weights": [0.2, 0.8],
    "max_context_chars": int(os.environ.get("RAG_MAX_CONTEXT", "3000")),
}

# =============================================================================
# CONFIGURATION EMBEDDINGS
# =============================================================================

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "nomic-embed-text")

# =============================================================================
# HELPERS
# =============================================================================

def get_active_model_config() -> dict:
    """Récupère la configuration du modèle actif (RAG_MODEL surcharge tout)."""
    override = os.environ.get("RAG_MODEL")
    if override:
        base_model = MODELS.get(ACTIVE_MODEL, MODELS[DEFAULT_ACTIVE_MODEL])
        config = dict(base_model)
        config["name"] = override
        return config

    if ACTIVE_MODEL not in MODELS:
        raise ValueError(
            f"Modèle '{ACTIVE_MODEL}' non reconnu. Options: {list(MODELS.keys())}"
        )
    config = dict(MODELS[ACTIVE_MODEL])
    return config

def list_available_models() -> None:
    """Affiche les modèles disponibles avec leurs descriptions."""
    print("\nMODÈLES DISPONIBLES :\n")
    for key, model in MODELS.items():
        active = "(ACTIF)" if key == ACTIVE_MODEL else ""
        print(f"  {key:15} - {model['description']} ({model['ram_required']}) {active}")
    print()

if __name__ == "__main__":
    list_available_models()
    print(f"Modèle actif       : {get_active_model_config()}")
    print(f"Embeddings         : {EMBEDDING_MODEL}")
    print(f"ChromaDB           : {CHROMA_DB_PATH}")
    print(f"Données ingestion  : {DATA_PATH}")
    print(f"Retrieval          : {RAG_CONFIG}")
