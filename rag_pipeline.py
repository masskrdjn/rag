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

# Import optimization modules
from reranker import RAGRerankerLightweight as RAGReranker
from cache_manager import CacheManager
from query_expander import QueryExpander
from prompt_builder import PromptBuilder
from hallucination_detector import HallucinationDetectorLightweight as HallucinationDetector

class SimpleRAG:
    def __init__(self, 
                 retrieval_mode="similarity",
                 top_k=4,
                 score_threshold=0.6,
                 use_hybrid=True,
                 hybrid_weights=None,
                 model_name="mistral:7b",
                 max_context_chars=3000):
        """
        Initialise le pipeline RAG optimisé.
        """
        self.persist_directory = "/home/rag/chroma_db"
        self.model_name = model_name
        self.embedding_model = "nomic-embed-text"
        self.qa_chain = None
        self.max_context_chars = max_context_chars
        
        # Configuration de récupération
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_hybrid = use_hybrid
        self.hybrid_weights = hybrid_weights if hybrid_weights else [0.2, 0.8]
        
        # Cache pour les documents (nécessaire pour BM25)
        self.documents = None
        
        # Initialisation des modules d'optimisation
        print("🚀 Initialisation des modules d'optimisation...")
        self.cache_manager = CacheManager()
        self.reranker = RAGReranker()
        self.query_expander = QueryExpander(model_name=model_name)
        self.prompt_builder = PromptBuilder()
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
            
        # Initialiser le LLM (num_predict limite les tokens générés)
        self.llm = ChatOllama(model=self.model_name, temperature=0.1, num_predict=512)
        
        # Marqueur que setup a été fait
        self.qa_chain = True
    
    def _create_retriever(self, vectorstore):
        """Créer un récupérateur basé sur la configuration."""
        
        # Stocker le vectorstore
        self.vectorstore = vectorstore
        
        if self.use_hybrid:
            return self._create_hybrid_retriever(vectorstore)
        
        elif self.retrieval_mode == "similarity_score_threshold":
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k * 3, # Fetch more for reranking
                    "score_threshold": self.score_threshold
                }
            )
        
        else:
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k * 3} # Fetch more for reranking
            )
    
    def _create_hybrid_retriever(self, vectorstore):
        """
        Créer un récupérateur hybride BM25-first.
        Stratégie: BM25 d'abord (mots-clés exacts), puis fallback vectoriel.
        """
        
        # Charger les documents pour BM25
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
                print("Avertissement : Aucun document trouvé pour BM25")
                return vectorstore.as_retriever(search_kwargs={"k": self.top_k * 3})
        
        # Créer le récupérateur BM25
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.top_k * 3 # Fetch more for reranking
        
        # Stocker pour usage dans custom retriever
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k * 3}
        )
        
        return self._create_bm25_first_retriever()
    
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
        Génère une réponse à partir d'une question avec pipeline optimisé.
        """
        start_time = time.time()
        
        if not self.qa_chain:
            self.setup_chain()
            
        # 1. Vérifier le cache
        cached_result = self.cache_manager.get(question, self.top_k)
        if cached_result:
            if return_sources:
                return cached_result
            return cached_result['answer']
            
        # Ajustement dynamique du top_k
        current_k = self.top_k
        if dynamic_k:
            current_k = self._estimate_dynamic_topk(question)
            self._update_retriever_topk(current_k * 3) # Fetch 3x for reranking
            print(f"Dynamic top_k: {current_k} (fetching {current_k*3} candidates)")
        
        # 2. Expansion de la requête
        query_variants = self.query_expander.get_all_variants(question)
        print(f"Query variants: {query_variants}")
        
        # 3. Récupération multi-requêtes
        all_docs = []
        seen_contents = set()
        
        # Chercher avec la question originale
        docs = self.retriever.invoke(question)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
                
        # Chercher avec les variantes (limité)
        for variant in query_variants[:2]: # Max 2 variantes pour performance
            if variant != question:
                variant_docs = self.retriever.invoke(variant)
                for doc in variant_docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)
        
        # Convertir en format dict pour reranker
        docs_for_reranking = []
        for doc in all_docs:
            doc_dict = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                **doc.metadata
            }
            docs_for_reranking.append(doc_dict)
            
        # 4. Reranking
        reranked_docs = self.reranker.rerank(question, docs_for_reranking, top_k=current_k)
        
        # 5. Construction du prompt
        prompts = self.prompt_builder.build_full_prompt(question, reranked_docs)
        
        # 6. Génération
        response = self.llm.invoke(prompts['full']).content
        
        # 7. Détection d'hallucinations
        hallucination_check = self.hallucination_detector.check_hallucinations(
            response, reranked_docs
        )
        
        # Ajouter badge de confiance
        final_response = self.hallucination_detector.add_confidence_badge(
            response, hallucination_check
        )
        
        # Restaurer le k original
        if dynamic_k:
            self._update_retriever_topk(self.top_k * 3)
        
        # Préparer le résultat complet
        sources = []
        for doc in reranked_docs:
            source_url = doc.get('source_url', '')
            sources.append({
                'source_url': source_url if source_url else None,
                'title': doc.get('section_title', '') or doc.get('filename', 'Unknown').replace('.html', '').replace('_', ' '),
                'category': doc.get('category', 'Unknown'),
                'post_id': doc.get('post_id', ''),
                'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
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
                'hallucination_check': hallucination_check
            }
        }
        
        # 8. Mise en cache
        self.cache_manager.set(question, self.top_k, result)
        
        if not return_sources:
            return final_response
            
        return result
    
    def _estimate_dynamic_topk(self, question):
        """Estime le top_k optimal selon la complexité de la question."""
        complexity_keywords = [
            "comment", "procédure", "étapes", "guide", "tuto", 
            "explication", "fonctionnement", "différence", "liste",
            "tous", "toutes", "quels", "quelles"
        ]
        
        precision_keywords = [
            "code", "format", "téléphone", "mail", "contact", 
            "adresse", "date", "prix", "montant"
        ]
        
        question_lower = question.lower()
        score = 0
        
        # Facteur longueur
        words = question.split()
        if len(words) > 10:
            score += 2
        elif len(words) > 5:
            score += 1
            
        # Facteur complexité
        if any(kw in question_lower for kw in complexity_keywords):
            score += 3
            
        # Facteur connecteurs
        if any(c in question_lower for c in [" et ", " ou ", " puis ", " ensuite "]):
            score += 2
            
        # Facteur précision
        if any(kw in question_lower for kw in precision_keywords):
            score -= 2
            
        return max(3, min(15, 5 + score))

    def _update_retriever_topk(self, k):
        """Met à jour le k du retriever."""
        if hasattr(self, '_bm25_retriever'):
            self._bm25_retriever.k = k
        if hasattr(self, '_vector_retriever') and hasattr(self._vector_retriever, 'search_kwargs'):
            self._vector_retriever.search_kwargs["k"] = k
    
    def _format_docs_for_context(self, docs):
        """Formate les documents pour le contexte."""
        # Note: This is now largely handled by PromptBuilder, but kept for compatibility if needed
        if not docs:
            return "Aucun contexte pertinent trouvé."
        
        return "\n\n".join([d.page_content for d in docs])

