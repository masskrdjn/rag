# config.py
"""
Configuration centrale pour le système RAG
"""

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

# Modèle actif (changer cette valeur pour basculer)
ACTIVE_MODEL = "qwen-14b"  # Options: mistral-7b, qwen-14b, qwen-7b, command-r-35b, llama-8b

# =============================================================================
# CONFIGURATION RAG
# =============================================================================

RAG_CONFIG = {
    "retrieval_mode": "similarity",
    "top_k": 6,  # Augmenté de 4 à 6 pour meilleur recall
    "score_threshold": 1.5,  # Augmenté de 0.5 à 1.5: distances ChromaDB peuvent aller jusqu'à ~0.8 pour docs pertinents
    "use_hybrid": True,
    "hybrid_weights": [0.2, 0.8],
    "max_context_chars": 3000,
}

# =============================================================================
# CONFIGURATION EMBEDDINGS
# =============================================================================

EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_PATH = "/home/rag/chroma_db"

# =============================================================================
# HELPERS
# =============================================================================

def get_active_model_config():
    """Récupère la configuration du modèle actif"""
    if ACTIVE_MODEL not in MODELS:
        raise ValueError(f"Modèle '{ACTIVE_MODEL}' non reconnu. Options: {list(MODELS.keys())}")
    return MODELS[ACTIVE_MODEL]

def list_available_models():
    """Liste tous les modèles disponibles avec leurs descriptions"""
    print("\n📋 MODÈLES DISPONIBLES:\n")
    for key, model in MODELS.items():
        active = "✓ ACTIF" if key == ACTIVE_MODEL else ""
        print(f"  {key:15} - {model['description']} ({model['ram_required']}) {active}")
    print()

if __name__ == "__main__":
    list_available_models()
    print(f"Configuration active: {get_active_model_config()}")
