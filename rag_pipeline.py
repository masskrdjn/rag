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
        Initialise le pipeline RAG.
        
        Args:
            retrieval_mode (str): "similarity" ou "similarity_score_threshold"
            top_k (int): Nombre de documents à récupérer (3-8 recommandé)
            score_threshold (float): Score de similarité minimum (0.5-0.8)
            use_hybrid (bool): Utiliser la recherche hybride BM25-first (recommandé)
            hybrid_weights (list): Non utilisé (conservé pour compatibilité)
            model_name (str): Modèle LLM à utiliser (mistral:7b, llama3.2, etc.)
            max_context_chars (int): Limite de caractères pour le contexte
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
                    "k": self.top_k,
                    "score_threshold": self.score_threshold
                }
            )
        
        else:
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
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
                return vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Créer le récupérateur BM25
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.top_k
        
        # Stocker pour usage dans custom retriever
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
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
                
                if relevant_docs:
                    return relevant_docs[:parent.top_k]
                
                # Fallback: recherche vectorielle
                return parent._vector_retriever.invoke(query)[:parent.top_k]
        
        return BM25FirstRetriever()

    def ask(self, question, return_sources=False, dynamic_k=True):
        """
        Génère une réponse à partir d'une question.
        
        Args:
            question: La question à poser
            return_sources: Si True, retourne aussi les sources
            dynamic_k: Si True, ajuste le top_k selon la complexité
        
        Returns:
            str ou dict avec 'answer', 'sources', 'metadata'
        """
        if not self.qa_chain:
            self.setup_chain()
            
        # Ajustement dynamique du top_k
        current_k = self.top_k
        if dynamic_k:
            current_k = self._estimate_dynamic_topk(question)
            self._update_retriever_topk(current_k)
            print(f"Dynamic top_k: {current_k} (base: {self.top_k})")
        
        # Récupérer les documents
        docs = self.retriever.invoke(question)
        
        # Formater le contexte
        context = self._format_docs_for_context(docs)
        
        # Limiter la taille du contexte
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n[... tronqué]"
        
        # Prompt optimisé
        template = """Tu es un assistant QA pour agents de voyages. Réponds uniquement avec le CONTEXTE fourni.

CONTEXTE:
{context}

QUESTION: {question}

Règles:
- Utilise UNIQUEMENT le contexte ci-dessus
- Si non trouvé: "Je n'ai pas trouvé cette information."
- Sois concis et précis

RÉPONSE:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "question": question})
        
        # Restaurer le k original
        if dynamic_k:
            self._update_retriever_topk(self.top_k)
        
        if not return_sources:
            return response
        
        # Préparer les sources
        sources = []
        for doc in docs:
            source_url = doc.metadata.get('source_url', '')
            sources.append({
                'source_url': source_url if source_url else None,
                'title': doc.metadata.get('section_title', '') or doc.metadata.get('filename', 'Unknown').replace('.html', '').replace('_', ' '),
                'category': doc.metadata.get('category', 'Unknown'),
                'post_id': doc.metadata.get('post_id', ''),
                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            })
        
        return {
            'answer': response,
            'sources': sources,
            'metadata': {
                'top_k': current_k,
                'dynamic_k_used': dynamic_k,
                'retrieval_mode': self.retrieval_mode,
                'use_hybrid': self.use_hybrid,
                'num_sources': len(sources)
            }
        }
    
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
        if not docs:
            return "Aucun contexte pertinent trouvé."
        
        # Regrouper par source
        by_source = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = {
                    'docs': [],
                    'filename': doc.metadata.get('filename', '')
                }
            by_source[source]['docs'].append(doc)
        
        formatted_parts = []
        
        for source, data in by_source.items():
            source_docs = data['docs']
            doc_title = data['filename'].replace('.html', '').replace('_', ' ')
            
            # Trier par section_num
            source_docs.sort(key=lambda d: (
                0 if d.metadata.get('is_summary', False) else 1,
                d.metadata.get('section_num', 999)
            ))
            
            source_content = []
            if len(by_source) > 1:
                source_content.append(f"=== {doc_title} ===")
            
            for doc in source_docs:
                section_title = doc.metadata.get('section_title', '')
                content = doc.page_content.strip()
                
                if section_title:
                    source_content.append(f"[{section_title}]")
                source_content.append(content)
            
            formatted_parts.append("\n\n".join(source_content))
        
        return "\n\n---\n\n".join(formatted_parts)
