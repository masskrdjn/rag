from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from rag_pipeline import SimpleRAG
import uvicorn
import os

# Initialize FastAPI
app = FastAPI(title="RAG Enterprise API", version="1.0.0")

# Initialize RAG Pipeline (Global instance)
# In a production setting, you might want to load this lazily or handle concurrency better
rag = SimpleRAG()

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

