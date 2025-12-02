import os
# Force CPU usage to avoid CUDA compatibility issues with old GPU (GT 1030)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag_pipeline import SimpleRAG
import uvicorn

# Initialize FastAPI
app = FastAPI(title="RAG Enterprise API", version="1.0.0")

# Configuration via variables d'environnement
MODEL_NAME = os.getenv("RAG_MODEL", "mistral:7b")
MAX_CONTEXT = int(os.getenv("RAG_MAX_CONTEXT", "3000"))

print(f"🚀 Configuration RAG: model={MODEL_NAME}, max_context={MAX_CONTEXT}")

# Initialize RAG Pipeline (Global instance)
rag = SimpleRAG(
    model_name=MODEL_NAME,
    max_context_chars=MAX_CONTEXT
)

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

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup."""
    print("Starting up RAG API...")
    try:
        rag.setup_chain()
        print("RAG Pipeline ready.")
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        # We don't exit here to allow the server to start, but requests will fail
        # In prod, you might want to exit or health-check fail

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Endpoint to ask a question to the RAG system."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Demander la réponse avec les sources
        result = rag.ask(request.question, return_sources=True, dynamic_k=request.dynamic_k)
        
        # Si le résultat est une string (ancien comportement), le convertir
        if isinstance(result, str):
            return QueryResponse(answer=result)
        
        # Convertir les sources au format SourceInfo
        sources = [SourceInfo(**s) for s in result.get('sources', [])]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            metadata=result.get('metadata', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run with: python server.py
    # Or in prod: uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
    uvicorn.run(app, host="0.0.0.0", port=8000)

