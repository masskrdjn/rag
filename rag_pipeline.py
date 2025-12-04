from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import os
import re
import time

# Import configuration
from config import get_active_model_config, RAG_CONFIG, EMBEDDING_MODEL, CHROMA_DB_PATH

from langchain_core.retrievers import BaseRetriever
from typing import Dict, List
# Import optimization modules
from reranker import RAGRerankerStrict as RAGReranker
from cache_manager import CacheManager
from query_expander import QueryExpander
from hallucination_detector import HallucinationDetectorLightweight as HallucinationDetector

class SimpleRAG:
    def __init__(self, 
                 retrieval_mode=None,
                 top_k=None,
                 score_threshold=None,
                 use_hybrid=None,
                 hybrid_weights=None,
                 model_name=None,
                 max_context_chars=None):
        """
        Initialise le pipeline RAG optimisé avec configuration centralisée.
        """
        # Charger la config du modèle actif
        model_config = get_active_model_config()
        
        # Paramètres du modèle
        self.model_name = model_name or model_config["name"]
        self.model_temperature = model_config["temperature"]
        self.model_num_predict = model_config["num_predict"]
        
        # Paramètres RAG (utilise config ou valeurs par défaut)
        self.persist_directory = CHROMA_DB_PATH
        self.embedding_model = EMBEDDING_MODEL
        self.max_context_chars = max_context_chars or RAG_CONFIG["max_context_chars"]
        
        # Configuration de récupération
        self.retrieval_mode = retrieval_mode or RAG_CONFIG["retrieval_mode"]
        self.top_k = top_k or RAG_CONFIG["top_k"]
        self.score_threshold = score_threshold or RAG_CONFIG["score_threshold"]
        self.use_hybrid = use_hybrid if use_hybrid is not None else RAG_CONFIG["use_hybrid"]
        self.hybrid_weights = hybrid_weights or RAG_CONFIG["hybrid_weights"]
        
        # Cache pour les documents (nécessaire pour BM25)
        self.documents = None
        self.qa_chain = None
        
        # Initialisation des modules d'optimisation
        print(f"🚀 Initialisation avec modèle: {self.model_name}")
        self.cache_manager = CacheManager()
        self.reranker = RAGReranker()
        self.query_expander = QueryExpander(model_name=self.model_name)
        self.hallucination_detector = HallucinationDetector()
        print("✓ Modules chargés")

    def setup_chain(self):
        """Initialise le pipeline RAG."""
        # Initialiser les embeddings
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Initialiser le magasin vectoriel
        vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # Configurer le récupérateur selon le mode
        self.retriever = self._create_retriever(vectorstore)
            
        # Initialiser le LLM avec paramètres de config
        self.llm = ChatOllama(
            model=self.model_name, 
            temperature=self.model_temperature, 
            num_predict=self.model_num_predict
        )
        
        # Marqueur que setup a été fait
        self.qa_chain = True
    
    def _create_retriever(self, vectorstore):
        """Créer un récupérateur optimisé avec score threshold."""
        
        self.vectorstore = vectorstore
        
        if self.use_hybrid:
            return self._create_hybrid_retriever_smart(vectorstore)
        elif self.retrieval_mode == "similarity_score_threshold":
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": min(self.top_k * 4, 10),  # Cap à 10
                    "score_threshold": 0.55  # ← STRICT!
                }
            )
        else:
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": min(self.top_k * 4, 10)}
            )
    
    def _create_hybrid_retriever_smart(self, vectorstore):
        """
        Hybrid retriever optimisé:
        - Vectoriel d'abord (sémantique fiable)
        - BM25 en complément si peu de résultats
        - Score threshold strict
        - Déduplication
        """
        
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": min(self.top_k * 4, 10),  # Cap à 10 candidates
            }
        )
        
        # BM25 comme backup
        if self.documents is None:
            all_docs = vectorstore.get()
            if all_docs and 'documents' in all_docs and all_docs['documents']:
                self.documents = [
                    Document(page_content=doc, metadata=meta if meta else {})
                    for doc, meta in zip(
                        all_docs['documents'],
                        all_docs['metadatas'] if 'metadatas' in all_docs else [{}] * len(all_docs['documents'])
                    )
                ]
            else:
                return vector_retriever  # Fallback si pas de docs
        
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = min(self.top_k * 2, 6)  # Cap BM25 à 6
        
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        
        parent = self
        
        class HybridRetrieverSmart(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # Phase 1: Vectoriel
                vector_docs = parent._vector_retriever.invoke(query)
                
                # Phase 2: BM25 - activé si peu de résultats OU mots-clés GDS
                query_lower = query.lower()
                gds_keywords = ['gds', 'galileo', 'sabre', 'amadeus', 'émettre', 'émission']
                force_bm25 = any(kw in query_lower for kw in gds_keywords)
                
                if len(vector_docs) < 3 or force_bm25:
                    reason = "GDS keywords detected" if force_bm25 else f"vector_docs={len(vector_docs)}"
                    print(f"🔍 BM25 activated ({reason})")
                    bm25_docs = parent._bm25_retriever.invoke(query)
                else:
                    print(f"✓ BM25 skipped (vector_docs={len(vector_docs)} sufficient)")
                    bm25_docs = []
                
                # Phase 3: Déduplication (premiers 200 chars)
                seen = {}
                combined = []
                
                for doc in vector_docs:
                    key = doc.page_content[:200]
                    if key not in seen:
                        combined.append(doc)
                        seen[key] = True
                
                for doc in bm25_docs:
                    key = doc.page_content[:200]
                    if key not in seen:
                        combined.append(doc)
                        seen[key] = True
                
                # Phase 4: Limit final (augmenté pour top_k=6)
                return combined[:min(parent.top_k * 2, 12)]  # Max 12 (au lieu de 6)
        
        return HybridRetrieverSmart()
    
    def _create_bm25_first_retriever(self):
        """
        Crée un retriever BM25-first avec extraction de mots-clés.
        """
        from langchain_core.retrievers import BaseRetriever
        from typing import List
        
        parent = self
        
        # Mots vides français à ignorer
        STOP_WORDS = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
            'me', 'te', 'se', 'lui', 'y', 'en',
            'et', 'ou', 'mais', 'donc', 'car', 'ni', 'que', 'qui', 'quoi',
            'comment', 'pourquoi', 'quand', 'où', 'quel', 'quelle', 'quels', 'quelles',
            'est', 'sont', 'être', 'avoir', 'fait', 'faire', 'peut', 'pouvoir',
            'pour', 'par', 'avec', 'sans', 'sur', 'sous', 'dans', 'entre',
            'à', 'a', 'ai', 'as', 'avez', 'avons', 'ont',
            'je', 'j', 'ne', 'pas', 'plus', 'moins', 'très', 'bien', 'mal',
            'tout', 'tous', 'toute', 'toutes', 'rien', 'quelque', 'chaque',
            'si', 'aussi', 'encore', 'déjà', 'toujours', 'jamais', 'ici', 'là',
            'faut', 'doit', 'dois', 'veut', 'veux', 'peux', 'suis', 'es'
        }
        
        class BM25FirstRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # Extraire les mots-clés significatifs (> 3 chars, pas stop words)
                words = re.findall(r'\b\w+\b', query.lower())
                keywords = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
                
                # Recherche BM25 avec les mots-clés
                if keywords:
                    keyword_query = ' '.join(keywords)
                    bm25_docs = parent._bm25_retriever.invoke(keyword_query)
                else:
                    bm25_docs = parent._bm25_retriever.invoke(query)
                
                # Vérifier si les résultats BM25 sont pertinents
                relevant_docs = []
                for doc in bm25_docs:
                    content_lower = doc.page_content.lower()
                    if any(kw in content_lower for kw in keywords):
                        relevant_docs.append(doc)
                
                # Si BM25 donne de bons résultats, on les garde
                if len(relevant_docs) >= 3:
                    return relevant_docs
                
                # Sinon, on complète avec la recherche vectorielle
                vector_docs = parent._vector_retriever.invoke(query)
                
                # Combiner et dédupliquer (par contenu)
                seen_content = set(d.page_content for d in relevant_docs)
                for doc in vector_docs:
                    if doc.page_content not in seen_content:
                        relevant_docs.append(doc)
                        seen_content.add(doc.page_content)
                
                return relevant_docs
        
        return BM25FirstRetriever()

    def ask(self, question, return_sources=False, dynamic_k=True):
        """
        Génération avec détection ACTIVE et rejection des hallucinations.
        Cache sélectif: ONLY good answers cached.
        """
        start_time = time.time()
        
        if not self.qa_chain:
            self.setup_chain()
        
        print(f"🚀 DEBUG V2: ask() called for question: '{question}'")
        
        # Vérifier le cache (même logique qu'avant)
        cached_result = self.cache_manager.get(question, self.top_k)
        if cached_result:
            if return_sources:
                return cached_result
            return cached_result['answer']
        
        # Ajustement dynamique du top_k
        current_k = self.top_k
        if dynamic_k:
            current_k = self._estimate_dynamic_topk(question)
            self._update_retriever_topk(current_k * 3)
        
        # FIX: Smart expansion - désactiver si question contient mots-clés précis
        # Détection de questions avec termes spécifiques (ne nécessitent PAS d'expansion)
        question_lower = question.lower()
        
        # Mots-clés précis qui indiquent une question ciblée
        specific_keywords = [
            'couleur', 'robot', 'congé', 'gds', 'galileo', 'sabre',
            'bsp', 'whaller', 'tarif', 'émission', 'règle',
            'procédure', 'étape', 'format', 'code'
        ]
        
        has_specific_keywords = any(kw in question_lower for kw in specific_keywords)
        
        # Désactiver expansion si :
        # 1. Question contient des mots-clés précis (ex: "couleurs du robot")
        # 2. Question courte (< 10 mots) avec termes spécifiques
        word_count = len(question.split())
        disable_expansion = has_specific_keywords or (word_count <= 10 and any(kw in question_lower for kw in ['comment', 'quoi']))
        
        print(f"🔍 DEBUG: Question='{question}'")
        print(f"🔍 DEBUG: has_specific_keywords={has_specific_keywords}")
        print(f"🔍 DEBUG: disable_expansion={disable_expansion}")
        
        if disable_expansion:
            print(f"✓ DEBUG: Expansion DÉSACTIVÉE pour '{question}' (keywords précis détectés)")
            query_variants = []
            num_variants_to_use = 0
        else:
            # Questions vraiment vagues nécessitant expansion
            vague_keywords = ['que faire', 'problème', 'échec', 'erreur', 'impossible']
            is_very_vague = any(kw in question_lower for kw in vague_keywords)
            
            if is_very_vague:
                print(f"🔍 Question très vague détectée, expansion LLM activée")
                query_variants = self.query_expander.get_all_variants(question, use_llm=True, llm_variants=2)
                num_variants_to_use = 2
            else:
                print(f"✓ Expansion minimale (pas de LLM)")
                query_variants = self.query_expander.get_all_variants(question, use_llm=False)
                num_variants_to_use = 1
        
        if num_variants_to_use > 0:
            print(f"Query variants ({num_variants_to_use} utilisées): {query_variants[:num_variants_to_use]}")
        
        # Initialisation des documents récupérés
        all_docs = []
        seen_contents = set()
        
        # Requête originale (toujours exécutée)
        docs = self.retriever.invoke(question)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        # Récupération multi-variantes (variantes supplémentaires) - SEULEMENT si expansion activée
        if num_variants_to_use > 0:
            for variant in query_variants[:num_variants_to_use]:
                if variant != question:
                    variant_docs = self.retriever.invoke(variant)
                    for doc in variant_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
        
        # Converter en format dict pour reranker
        docs_for_reranking = []
        for doc in all_docs:
            doc_dict = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                **doc.metadata
            }
            docs_for_reranking.append(doc_dict)
        
        # Reranking (utilise RAGRerankerStrict maintenant)
        reranked_docs = self.reranker.rerank(question, docs_for_reranking, top_k=current_k)
        
        # ⚠️ CHECK: Avons-nous des sources?
        if len(reranked_docs) == 0:
            final_response = '❌ Aucun document pertinent trouvé pour cette question.'
            result = {
                'answer': final_response,
                'sources': [],
                'metadata': {
                    'error': 'NO_RELEVANT_DOCS',
                    'execution_time': time.time() - start_time
                }
            }
            if return_sources:
                return result
            return final_response
        
        # GÉNÉRATION avec constraint
        context = self._format_docs_for_context(reranked_docs)
        prompt_with_constraints = f"""Réponds en utilisant UNIQUEMENT les informations des sources fournies ci-dessous.

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
        
        response = self.llm.invoke(prompt_with_constraints).content
        
        # DÉTECTION hallucinations
        hallucination_check = self.hallucination_detector.check_hallucinations(
            response, reranked_docs
        )
        
        confidence = hallucination_check['confidence_score']
        
        # DÉCISION: Accept, Warn, or Reject?
        if confidence < 0.25:
            # ❌ REJECTION: Hallucinations sévères
            print(f"⚠️ REJECTION: confidence={confidence:.2f}")
            final_response = f"""Je ne peux pas répondre à cette question de manière fiable.

**Informations trouvées dans les documents:**
{self._extract_claims(hallucination_check, 'supported')}

**Pour plus de détails, consultez directement les sources ci-dessous.**"""
            should_cache = False
            
        elif confidence < 0.5:
            # ⚠️ WARNING: Quelques hallucinations
            print(f"⚠️ WARNING: confidence={confidence:.2f}")
            final_response = response + f"\n\n⚠️ **[CONFIANCE FAIBLE - {int(confidence*100)}%]**\nConsultez les sources pour confirmation."
            should_cache = False
            
        else:
            # ✅ ACCEPT: Réponse fiable
            final_response = response
            should_cache = True
        
        # Restaurer le k original
        if dynamic_k:
            self._update_retriever_topk(self.top_k * 3)
        
        # Préparer les sources
        sources = []
        for doc in reranked_docs:
            source_url = doc.get('source_url', '')
            sources.append({
                'source_url': source_url if source_url else None,
                'title': doc.get('section_title', '') or doc.get('filename', 'Unknown').replace('.html', '').replace('_', ' '),
                'category': doc.get('category', 'Unknown'),
                'post_id': doc.get('post_id', ''),
                'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', ''),
                'rerank_score': doc.get('rerank_score', 0)
            })
        
        result = {
            'answer': final_response,
            'sources': sources,
            'metadata': {
                'top_k': current_k,
                'dynamic_k_used': dynamic_k,
                'retrieval_mode': self.retrieval_mode,
                'use_hybrid': self.use_hybrid,
                'num_sources': len(sources),
                'execution_time': time.time() - start_time,
                'hallucination_check': hallucination_check,
                'confidence': confidence,
                'action': 'ACCEPT' if confidence > 0.5 else 'WARN' if confidence > 0.25 else 'REJECT'
            }
        }
        
        # Cache sélectif: ONLY good answers
        if should_cache:
            self.cache_manager.set(question, self.top_k, result)
        
        if not return_sources:
            return final_response
        
        return result

    def _extract_claims(self, hallucination_check: Dict, claim_type: str) -> str:
        """Extract formatted claims"""
        claims = hallucination_check.get(f'{claim_type}_claims', [])
        if not claims:
            return "Aucune information trouvée."
        return "\n".join([f"• {c}" for c in claims[:3]])
    
    def _estimate_dynamic_topk(self, question):
        """Estime le top_k optimal selon la complexité de la question."""
        # FIX 1: Force limit to 4 sources to reduce latency and noise
        return 4

    def _update_retriever_topk(self, k):
        """Met à jour le k du retriever."""
        if hasattr(self, '_bm25_retriever'):
            self._bm25_retriever.k = k
        if hasattr(self, '_vector_retriever') and hasattr(self._vector_retriever, 'search_kwargs'):
            self._vector_retriever.search_kwargs["k"] = k
    
    def _format_docs_for_context(self, docs):
        """Format documents for LLM context with metadata"""
        if not docs:
            return "Aucun contexte pertinent trouvé."
        
        formatted = []
        total_chars = 0
        max_total = self.max_context_chars  # 3000 par défaut
        
        for i, doc in enumerate(docs, 1):
            content = doc.get('content', '') or doc.get('page_content', '')
            title = doc.get('section_title', '') or doc.get('filename', 'Document')
            category = doc.get('category', '')
            
            # Calculer l'espace restant, minimum 1000 chars par doc (augmenté de 800)
            remaining = max_total - total_chars
            chars_per_doc = max(1000, remaining // (len(docs) - i + 1))
            
            # Tronquer intelligemment (à la fin d'une phrase si possible)
            if len(content) > chars_per_doc:
                # Chercher la dernière phrase complète
                truncated = content[:chars_per_doc]
                last_period = truncated.rfind('.')
                if last_period > chars_per_doc * 0.7:  # Si on trouve un point après 70%
                    truncated = truncated[:last_period + 1]
                content = truncated + "..."
            
            # Format enrichi avec métadonnées
            metadata_str = f" [Catégorie: {category}]" if category else ""
            formatted.append(f"[Source {i}: {title}]{metadata_str}\n{content}")
            total_chars += len(content)
        
        return "\n\n".join(formatted)

