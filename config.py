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

import json
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
        "num_predict": 1024,
    },
    "qwen-14b": {
        "name": "qwen2.5:14b",
        "description": "Qwen 2.5 14B - Meilleur suivi d'instructions, excellent français",
        "ram_required": "10-12GB",
        "temperature": 0.1,
        "num_predict": 1024,
    },
    "qwen-7b": {
        "name": "qwen2.5:7b",
        "description": "Qwen 2.5 7B - Bon compromis, meilleur que mistral",
        "ram_required": "6-8GB",
        "temperature": 0.1,
        "num_predict": 1024,
    },
    "command-r-35b": {
        "name": "command-r:35b",
        "description": "Command-R 35B - Optimisé pour RAG, citations précises",
        "ram_required": "20-24GB",
        "temperature": 0.1,
        "num_predict": 1024,
    },
    "qwen3-14b": {
        "name": "qwen3:14b",
        "description": "Qwen 3 14B - Profil optionnel a benchmarker pour RAG FR",
        "ram_required": "10-14GB",
        "temperature": 0.1,
        "num_predict": 1024,
    },
    "mistral-small3.2": {
        "name": "mistral-small3.2:latest",
        "description": "Mistral Small 3.2 - Profil optionnel recent, bon francais",
        "ram_required": "14-20GB",
        "temperature": 0.1,
        "num_predict": 1024,
    },
    "llama-8b": {
        "name": "llama3.1:8b",
        "description": "Llama 3.1 8B - Rapide et fiable",
        "ram_required": "6-8GB",
        "temperature": 0.1,
        "num_predict": 1024,
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
    - Linux production : /home/rag/chroma_db si la base existe vraiment
    - Sinon (Windows, dev local) : <projet>/chroma_db
    """
    linux_default = Path("/home/rag/chroma_db")
    if linux_default.exists():
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


def _env_optional_int(key: str, default: int = None) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


DEFAULT_KEYWORD_TAXONOMY = {
    "gds": [
        "gds", "galileo", "sabre", "amadeus", "bsp", "pnr",
        "billet", "emission", "emettre", "reemettre", "void",
    ],
    "specific": [
        "couleur", "robot", "conge", "gds", "galileo", "sabre",
        "bsp", "whaller", "tarif", "emission", "regle", "procedure",
        "etape", "format", "code", "train", "lowcost", "sncf",
    ],
    "vague": ["que faire", "probleme", "echec", "erreur", "impossible"],
    "broad": [
        "comment", "procedure", "process", "etapes", "resume",
        "explique", "guide", "que faire",
    ],
}


def _load_keyword_taxonomy() -> dict:
    """
    Charge une taxonomie JSON optionnelle sans ajouter de dependance YAML.
    Le fichier doit contenir un objet {"gds": [...], "specific": [...], ...}.
    """
    config_path = os.environ.get("RAG_KEYWORDS_CONFIG")
    if not config_path:
        return DEFAULT_KEYWORD_TAXONOMY

    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        taxonomy = dict(DEFAULT_KEYWORD_TAXONOMY)
        for key, value in loaded.items():
            if isinstance(value, list):
                taxonomy[key] = [str(item) for item in value]
        return taxonomy
    except Exception as exc:
        print(f"Taxonomie keywords ignoree ({path}): {exc}")
        return DEFAULT_KEYWORD_TAXONOMY


RAG_CONFIG = {
    "retrieval_mode": "similarity",
    "top_k": int(os.environ.get("RAG_TOP_K", "6")),
    "max_dynamic_top_k": int(os.environ.get("RAG_MAX_DYNAMIC_TOP_K", "8")),
    "score_threshold": 1.5,
    "use_hybrid": _env_bool("RAG_USE_HYBRID", True),
    # [BM25, vector] — corpus métier riche en acronymes (GDS, BSP, PNR…)
    # nécessite un BM25 fort. Tester [0.4, 0.6] ou [0.6, 0.4] sur ton batch.
    "hybrid_weights": [0.5, 0.5],
    "max_context_chars": int(os.environ.get("RAG_MAX_CONTEXT", "3000")),
    "max_question_chars": int(os.environ.get("RAG_MAX_QUESTION_CHARS", "1000")),
    "cache_ttl_seconds": _env_optional_int("RAG_CACHE_TTL_SECONDS"),
    "corpus_build_id": os.environ.get("RAG_CORPUS_BUILD_ID", ""),
    "api_key": os.environ.get("RAG_API_KEY", ""),
    "keyword_taxonomy": _load_keyword_taxonomy(),
}

# =============================================================================
# CONFIGURATION EMBEDDINGS
# =============================================================================

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "nomic-embed-text")
RECOMMENDED_EMBEDDING_MODELS = {
    "default": "nomic-embed-text",
    "french_benchmark": "bge-m3",
}

# =============================================================================
# HELPERS
# =============================================================================

def get_corpus_build_id(chroma_path: str = None) -> str:
    """
    Identifiant leger pour invalider le cache apres reingestion.
    RAG_CORPUS_BUILD_ID reste prioritaire pour la prod.
    """
    explicit = RAG_CONFIG.get("corpus_build_id")
    if explicit:
        return explicit

    path = Path(chroma_path or CHROMA_DB_PATH)
    if not path.exists():
        return "missing-corpus"

    candidates = [path / "chroma.sqlite3", path]
    parts = []
    for candidate in candidates:
        if candidate.exists():
            stat = candidate.stat()
            parts.append(f"{candidate.name}:{int(stat.st_mtime)}:{stat.st_size}")
    return "|".join(parts) or str(path)

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
    print(f"Embeddings option  : {RECOMMENDED_EMBEDDING_MODELS}")
    print(f"ChromaDB           : {CHROMA_DB_PATH}")
    print(f"Corpus build id    : {get_corpus_build_id()}")
    print(f"Données ingestion  : {DATA_PATH}")
    print(f"Retrieval          : {RAG_CONFIG}")
