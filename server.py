"""
API FastAPI exposant le pipeline RAG.

Variables d'environnement utiles :
- RAG_DISABLE_CUDA : "1" pour forcer la désactivation de CUDA côté API
                     (utile sur GPU anciens incompatibles avec PyTorch récent).
                     Défaut : non défini = on respecte RAG_USE_GPU / device_config.
- Tous les autres RAG_* sont gérés par config.py.
"""

import os

# Désactivation conditionnelle de CUDA. Historiquement forcé pour contourner
# une incompatibilité avec une GT 1030 ; rendu optionnel pour les machines
# récentes.
if os.environ.get("RAG_DISABLE_CUDA", "0").lower() in ("1", "true", "yes"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import RAG_CONFIG, get_active_model_config
from rag_pipeline import SimpleRAG

_model_config = get_active_model_config()
MODEL_NAME = _model_config["name"]
MAX_CONTEXT = RAG_CONFIG["max_context_chars"]

print(f"Configuration RAG : model={MODEL_NAME}, max_context={MAX_CONTEXT}")

rag = SimpleRAG(model_name=MODEL_NAME, max_context_chars=MAX_CONTEXT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarrage de l'API RAG...")
    try:
        rag.setup_chain()
        print("Pipeline RAG prêt.")
    except Exception as e:
        # On ne lève pas pour permettre au serveur de démarrer ;
        # /health restera up mais /ask renverra 500.
        print(f"Échec d'initialisation du pipeline : {e}")
    yield


app = FastAPI(title="RAG Enterprise API", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    dynamic_k: bool = True


class SourceInfo(BaseModel):
    source_url: Optional[str] = None
    title: str = ""
    category: str = "Unknown"
    post_id: str = ""
    content_preview: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []
    metadata: Dict[str, Any] = {}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag.ask(
            request.question,
            return_sources=True,
            dynamic_k=request.dynamic_k,
        )
        if isinstance(result, str):
            return QueryResponse(answer=result)

        sources = [SourceInfo(**s) for s in result.get("sources", [])]
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
