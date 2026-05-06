"""
Factory d'embeddings Ollama avec préfixage adapté au modèle.

Pourquoi : `nomic-embed-text` (et plusieurs autres modèles modernes) attendent
un préfixe d'instruction différent selon que le texte est une requête ou un
document indexé. Sans préfixe, on perd plusieurs points de NDCG gratuitement.

Usage :
    from embeddings_factory import get_embeddings, EMBEDDING_MODEL

    # Côté ingestion
    embeddings = get_embeddings(EMBEDDING_MODEL)
    Chroma.from_documents(documents=chunks, embedding=embeddings, ...)

    # Côté retrieval (même fonction — la classe gère les deux rôles)
    embeddings = get_embeddings(EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=..., embedding_function=embeddings)

IMPORTANT : si tu changes la stratégie de préfixage, il faut **réingérer**
toute la base : les vecteurs document/query ne sont comparables que s'ils ont
été produits avec la même paire de préfixes.
"""

from typing import List, Tuple

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


# Mapping modèle → (préfixe query, préfixe document).
# Sources :
# - nomic-embed-text v1.5+ : doc Nomic AI (https://blog.nomic.ai/posts/nomic-embed-text-v1)
# - bge-m3 : pas de préfixe nécessaire, alignement query/doc géré en interne
# - mxbai-embed-large : pas de préfixe par défaut
# Ajoute ici les modèles que tu veux supporter explicitement.
_PREFIX_TABLE: dict[str, Tuple[str, str]] = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


def _resolve_prefixes(model_name: str) -> Tuple[str, str]:
    """Retourne (query_prefix, doc_prefix) pour le modèle donné, ou ('', '')."""
    name = model_name.lower().split(":", 1)[0]  # drop tag, ex "nomic-embed-text:latest"
    for key, prefixes in _PREFIX_TABLE.items():
        if key in name:
            return prefixes
    return "", ""


class PrefixedOllamaEmbeddings(Embeddings):
    """
    Wrapper LangChain `Embeddings` qui préfixe les textes avant de déléguer à
    OllamaEmbeddings. Compatible avec Chroma comme `embedding_function`.
    """

    def __init__(self, model: str, query_prefix: str = "", document_prefix: str = ""):
        self._base = OllamaEmbeddings(model=model)
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self.model = model

    def embed_query(self, text: str) -> List[float]:
        return self._base.embed_query(
            f"{self._query_prefix}{text}" if self._query_prefix else text
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._document_prefix:
            texts = [f"{self._document_prefix}{t}" for t in texts]
        return self._base.embed_documents(texts)


def get_embeddings(model_name: str) -> Embeddings:
    """
    Construit une instance d'embeddings pour Ollama avec préfixage adapté.

    Si le modèle ne nécessite pas de préfixe, retourne directement
    `OllamaEmbeddings` (pas de surcoût).
    """
    query_prefix, document_prefix = _resolve_prefixes(model_name)
    if query_prefix or document_prefix:
        return PrefixedOllamaEmbeddings(
            model=model_name,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
    return OllamaEmbeddings(model=model_name)
