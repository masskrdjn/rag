"""
Pipeline RAG : retrieval hybride (vectoriel + BM25), reranking, expansion de
requête, détection d'hallucinations et génération via Ollama.
"""

import time
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings

from cache_manager import CacheManager
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, RAG_CONFIG, get_active_model_config
from device_config import get_device  # import déclenche l'affichage du statut device
from hallucination_detector import get_hallucination_detector
from query_expander import QueryExpander
from reranker import get_reranker

# Mots-clés métier déclenchant l'activation systématique de BM25
GDS_KEYWORDS = ("gds", "galileo", "sabre", "amadeus", "émettre", "émission")

# Mots-clés "précis" qui rendent l'expansion LLM contre-productive
SPECIFIC_KEYWORDS = (
    "couleur", "robot", "congé", "gds", "galileo", "sabre",
    "bsp", "whaller", "tarif", "émission", "règle",
    "procédure", "étape", "format", "code",
)

# Questions vraiment vagues nécessitant l'expansion LLM
VAGUE_KEYWORDS = ("que faire", "problème", "échec", "erreur", "impossible")


class SimpleRAG:
    def __init__(
        self,
        retrieval_mode: str = None,
        top_k: int = None,
        score_threshold: float = None,
        use_hybrid: bool = None,
        hybrid_weights=None,
        model_name: str = None,
        max_context_chars: int = None,
    ):
        model_config = get_active_model_config()

        self.model_name = model_name or model_config["name"]
        self.model_temperature = model_config["temperature"]
        self.model_num_predict = model_config["num_predict"]

        self.persist_directory = CHROMA_DB_PATH
        self.embedding_model = EMBEDDING_MODEL
        self.max_context_chars = max_context_chars or RAG_CONFIG["max_context_chars"]

        self.retrieval_mode = retrieval_mode or RAG_CONFIG["retrieval_mode"]
        self.top_k = top_k or RAG_CONFIG["top_k"]
        self.score_threshold = score_threshold or RAG_CONFIG["score_threshold"]
        self.use_hybrid = use_hybrid if use_hybrid is not None else RAG_CONFIG["use_hybrid"]
        self.hybrid_weights = hybrid_weights or RAG_CONFIG["hybrid_weights"]

        self.documents: List[Document] = None
        self.qa_chain = None

        print(f"Initialisation modèle : {self.model_name} | device : {get_device().upper()}")
        self.cache_manager = CacheManager()
        self.reranker = get_reranker()
        self.query_expander = QueryExpander(model_name=self.model_name)
        self.hallucination_detector = get_hallucination_detector()
        print("Modules d'optimisation chargés")

    # ------------------------------------------------------------------ setup
    def setup_chain(self):
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
        )
        self.retriever = self._create_retriever(vectorstore)
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.model_temperature,
            num_predict=self.model_num_predict,
        )
        self.qa_chain = True

    def _create_retriever(self, vectorstore):
        self.vectorstore = vectorstore

        if self.use_hybrid:
            return self._create_hybrid_retriever_smart(vectorstore)

        if self.retrieval_mode == "similarity_score_threshold":
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": min(self.top_k * 4, 10),
                    "score_threshold": 0.55,
                },
            )

        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(self.top_k * 4, 10)},
        )

    def _create_hybrid_retriever_smart(self, vectorstore):
        """
        Hybride sélectif : vectoriel d'abord, BM25 en complément si peu de
        résultats ou si la question contient un mot-clé GDS.
        """
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": min(self.top_k * 4, 10)},
        )

        if self.documents is None:
            all_docs = vectorstore.get()
            if all_docs and all_docs.get("documents"):
                metas = all_docs.get("metadatas") or [{}] * len(all_docs["documents"])
                self.documents = [
                    Document(page_content=doc, metadata=meta or {})
                    for doc, meta in zip(all_docs["documents"], metas)
                ]
            else:
                return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = min(self.top_k * 2, 6)

        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever

        parent = self

        class HybridRetrieverSmart(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                vector_docs = parent._vector_retriever.invoke(query)

                query_lower = query.lower()
                force_bm25 = any(kw in query_lower for kw in GDS_KEYWORDS)
                if len(vector_docs) < 3 or force_bm25:
                    reason = "GDS keywords" if force_bm25 else f"vector_docs={len(vector_docs)}"
                    print(f"BM25 activé ({reason})")
                    bm25_docs = parent._bm25_retriever.invoke(query)
                else:
                    bm25_docs = []

                # Déduplication par préfixe (premiers 200 chars)
                seen = set()
                combined: List[Document] = []
                for doc in list(vector_docs) + list(bm25_docs):
                    key = doc.page_content[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(doc)

                return combined[: min(parent.top_k * 2, 12)]

        return HybridRetrieverSmart()

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, return_sources: bool = False, dynamic_k: bool = True):
        start_time = time.time()

        if not self.qa_chain:
            self.setup_chain()

        cached_result = self.cache_manager.get(question, self.top_k)
        if cached_result:
            return cached_result if return_sources else cached_result["answer"]

        current_k = self.top_k
        if dynamic_k:
            current_k = self._estimate_dynamic_topk(question)
            self._update_retriever_topk(current_k * 3)

        query_variants, num_variants_to_use = self._build_query_variants(question)

        # Récupération multi-variantes avec déduplication
        all_docs: List[Document] = []
        seen_contents = set()

        for doc in self.retriever.invoke(question):
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)

        for variant in query_variants[:num_variants_to_use]:
            if variant == question:
                continue
            for doc in self.retriever.invoke(variant):
                if doc.page_content not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)

        # Reranking
        docs_for_reranking = [
            {"content": d.page_content, "metadata": d.metadata, **d.metadata}
            for d in all_docs
        ]
        reranked_docs = self.reranker.rerank(question, docs_for_reranking, top_k=current_k)

        if not reranked_docs:
            final_response = "Aucun document pertinent trouvé pour cette question."
            result = {
                "answer": final_response,
                "sources": [],
                "metadata": {
                    "error": "NO_RELEVANT_DOCS",
                    "execution_time": time.time() - start_time,
                },
            }
            return result if return_sources else final_response

        # Génération
        context = self._format_docs_for_context(reranked_docs)
        prompt = self._build_generation_prompt(context, question)
        response = self.llm.invoke(prompt).content

        # Détection d'hallucinations
        hallucination_check = self.hallucination_detector.check_hallucinations(
            response, reranked_docs
        )
        confidence = hallucination_check["confidence_score"]

        if confidence < 0.25:
            print(f"REJECT : confidence={confidence:.2f}")
            final_response = (
                "Je ne peux pas répondre à cette question de manière fiable.\n\n"
                "**Informations trouvées dans les documents :**\n"
                f"{self._extract_claims(hallucination_check, 'supported')}\n\n"
                "**Pour plus de détails, consultez directement les sources ci-dessous.**"
            )
            should_cache = False
            action = "REJECT"
        elif confidence < 0.5:
            print(f"WARN : confidence={confidence:.2f}")
            final_response = (
                f"{response}\n\n"
                f"**[CONFIANCE FAIBLE - {int(confidence * 100)}%]**\n"
                "Consultez les sources pour confirmation."
            )
            should_cache = False
            action = "WARN"
        else:
            final_response = response
            should_cache = True
            action = "ACCEPT"

        if dynamic_k:
            self._update_retriever_topk(self.top_k * 3)

        sources = []
        for doc in reranked_docs:
            content = doc.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            source_url = doc.get("source_url", "")
            sources.append({
                "source_url": source_url or None,
                "title": (
                    doc.get("section_title", "")
                    or doc.get("filename", "Unknown").replace(".html", "").replace("_", " ")
                ),
                "category": doc.get("category", "Unknown"),
                "post_id": doc.get("post_id", ""),
                "content_preview": preview,
                "rerank_score": doc.get("rerank_score", 0),
            })

        result = {
            "answer": final_response,
            "sources": sources,
            "metadata": {
                "top_k": current_k,
                "dynamic_k_used": dynamic_k,
                "retrieval_mode": self.retrieval_mode,
                "use_hybrid": self.use_hybrid,
                "num_sources": len(sources),
                "execution_time": time.time() - start_time,
                "hallucination_check": hallucination_check,
                "confidence": confidence,
                "action": action,
            },
        }

        if should_cache:
            self.cache_manager.set(question, self.top_k, result)

        return result if return_sources else final_response

    # ------------------------------------------------------------------ helpers
    def _build_query_variants(self, question: str):
        """
        Désactive l'expansion sur les questions ciblées (mots-clés précis).
        Active l'expansion LLM uniquement sur les questions vraiment vagues.
        """
        question_lower = question.lower()
        has_specific = any(kw in question_lower for kw in SPECIFIC_KEYWORDS)
        word_count = len(question.split())
        disable_expansion = has_specific or (
            word_count <= 10 and any(kw in question_lower for kw in ("comment", "quoi"))
        )

        if disable_expansion:
            return [], 0

        is_very_vague = any(kw in question_lower for kw in VAGUE_KEYWORDS)
        if is_very_vague:
            variants = self.query_expander.get_all_variants(
                question, use_llm=True, llm_variants=2
            )
            return variants, 2

        variants = self.query_expander.get_all_variants(question, use_llm=False)
        return variants, 1

    @staticmethod
    def _build_generation_prompt(context: str, question: str) -> str:
        return f"""Réponds en utilisant UNIQUEMENT les informations des sources fournies ci-dessous.

INSTRUCTIONS:
1. Utilise la source la plus pertinente comme base principale
2. Si d'autres sources traitent de cas différents, mentionne-le clairement
3. Pour les listes (couleurs, étapes), explique la signification de chaque élément
4. Si la réponse ne concerne qu'un cas spécifique (ex: "connecteur SNCF"), précise-le
5. Cite les sources: (Source X)

Si l'information n'est PAS dans les sources:
"Je n'ai pas trouvé cette information dans les documents disponibles."

SOURCES:
{context}

QUESTION: {question}

RÉPONSE:"""

    def _extract_claims(self, hallucination_check: Dict, claim_type: str) -> str:
        claims = hallucination_check.get(f"{claim_type}_claims", [])
        if not claims:
            return "Aucune information trouvée."
        return "\n".join([f"- {c}" for c in claims[:3]])

    def _estimate_dynamic_topk(self, question: str) -> int:
        """Limite à 4 sources pour réduire latence et bruit."""
        return 4

    def _update_retriever_topk(self, k: int) -> None:
        if hasattr(self, "_bm25_retriever"):
            self._bm25_retriever.k = k
        if hasattr(self, "_vector_retriever") and hasattr(self._vector_retriever, "search_kwargs"):
            self._vector_retriever.search_kwargs["k"] = k

    def _format_docs_for_context(self, docs: List[Dict]) -> str:
        if not docs:
            return "Aucun contexte pertinent trouvé."

        formatted = []
        total_chars = 0
        max_total = self.max_context_chars

        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "") or doc.get("page_content", "")
            title = doc.get("section_title", "") or doc.get("filename", "Document")
            category = doc.get("category", "")

            remaining = max_total - total_chars
            chars_per_doc = max(1000, remaining // max(len(docs) - i + 1, 1))

            if len(content) > chars_per_doc:
                truncated = content[:chars_per_doc]
                last_period = truncated.rfind(".")
                if last_period > chars_per_doc * 0.7:
                    truncated = truncated[: last_period + 1]
                content = truncated + "..."

            metadata_str = f" [Catégorie: {category}]" if category else ""
            formatted.append(f"[Source {i}: {title}]{metadata_str}\n{content}")
            total_chars += len(content)

        return "\n\n".join(formatted)
