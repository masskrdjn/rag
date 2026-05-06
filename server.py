"""
API FastAPI exposant le pipeline RAG.

Variables d'environnement utiles :
- RAG_DISABLE_CUDA : "1" pour forcer la désactivation de CUDA côté API
                     (utile sur GPU anciens incompatibles avec PyTorch récent).
                     Défaut : non défini = CPU, sauf opt-in GPU explicite.
- RAG_USE_GPU      : "1"/"true"/"gpu"/"cuda" pour autoriser CUDA côté API
                     si RAG_DISABLE_CUDA n'est pas défini.
- Tous les autres RAG_* sont gérés par config.py.
"""

import os
import asyncio

# Désactivation conditionnelle de CUDA. Historiquement forcé pour contourner
# une incompatibilité avec une GT 1030 ; le CPU reste le défaut sûr pour l'API.
disable_cuda = os.environ.get("RAG_DISABLE_CUDA")
gpu_requested = os.environ.get("RAG_USE_GPU", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
    "gpu",
    "cuda",
)
if disable_cuda is None:
    disable_cuda = "0" if gpu_requested else "1"

if disable_cuda.strip().lower() in ("1", "true", "yes", "on", "gpu", "cuda"):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config import RAG_CONFIG, get_active_model_config
from rag_pipeline import SimpleRAG

_model_config = get_active_model_config()
MODEL_NAME = _model_config["name"]
MAX_CONTEXT = RAG_CONFIG["max_context_chars"]

print(f"Configuration RAG : model={MODEL_NAME}, max_context={MAX_CONTEXT}")

# Instance créée dans le lifespan pour ne PAS déclencher l'init (Ollama,
# reranker, détecteur, query expander) au simple import de `server`.
rag: Optional[SimpleRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    print("Démarrage de l'API RAG...")
    try:
        rag = SimpleRAG(model_name=MODEL_NAME, max_context_chars=MAX_CONTEXT)
        rag.setup_chain()
        print("Pipeline RAG prêt.")
    except Exception as e:
        # On ne lève pas pour permettre au serveur de démarrer ;
        # /health restera up mais /ask renverra 500.
        print(f"Échec d'initialisation du pipeline : {e}")
    yield


app = FastAPI(title="RAG Enterprise API", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=RAG_CONFIG["max_question_chars"],
    )
    dynamic_k: bool = True


class SourceInfo(BaseModel):
    source_url: Optional[str] = None
    title: str = ""
    category: str = "Unknown"
    post_id: str = ""
    content_preview: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    expected_api_key = RAG_CONFIG.get("api_key")
    if expected_api_key and x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if rag is None:
        raise HTTPException(status_code=503, detail="Pipeline RAG non initialisé")

    try:
        result = await asyncio.to_thread(
            rag.ask,
            question,
            True,
            request.dynamic_k,
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
