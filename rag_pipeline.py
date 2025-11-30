from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import os

class SimpleRAG:
    def __init__(self, 
                 retrieval_mode="similarity",
                 top_k=5,
                 score_threshold=0.6,
                 use_hybrid=False,
                 hybrid_weights=None):
        """
        Initialise le pipeline RAG avec des stratégies de récupération configurables.
        
        Args:
            retrieval_mode (str): "similarity" ou "similarity_score_threshold"
            top_k (int): Nombre de documents à récupérer (3-8 recommandé)
            score_threshold (float): Score de similarité minimum (0.5-0.8, uniquement pour le mode seuil)
            use_hybrid (bool): Utiliser la recherche hybride BM25 + vectorielle
            hybrid_weights (list): Poids pour [vectoriel, bm25] (par défaut [0.5, 0.5])
        """
        self.persist_directory = "/home/rag/chroma_db"  # Base de données ChromaDB
        self.model_name = "llama3.2"
        self.embedding_model = "nomic-embed-text"
        self.qa_chain = None
        
        # Configuration de récupération (améliorée pour mieux capturer les sections courtes)
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k  # Augmenté de 3 à 5 par défaut
        self.score_threshold = score_threshold  # Réduit de 0.7 à 0.6 pour être moins strict
        self.use_hybrid = use_hybrid
        self.hybrid_weights = hybrid_weights if hybrid_weights else [0.5, 0.5]
        
        # Cache pour les documents (nécessaire pour BM25)
        self.documents = None

    def setup_chain(self):
        # Initialiser les embeddings
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Initialiser le magasin vectoriel
        vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # Configurer le récupérateur selon le mode
        self.retriever = self._create_retriever(vectorstore)
        retriever = self.retriever
            
        # Initialiser le LLM avec température basse pour des réponses plus fidèles
        llm = ChatOllama(model=self.model_name, temperature=0.1)
        
        # Créer le modèle de prompt
        template = """Tu es un assistant QA RAG pour une documentation interne d'agents de voyages.

CONTEXTE RÉCUPÉRÉ :
{context}

QUESTION :
{question}

INSTRUCTIONS :
1. Réponds en te basant UNIQUEMENT sur le CONTEXTE RÉCUPÉRÉ ci-dessus.
2. Si le contexte mentionne de "consulter" une source externe (Whaller, sphère, intranet, fiche Gravity, etc.), 
   CITE cette source EXACTEMENT comme elle apparaît dans le contexte.
3. N'invente RIEN. Si l'information n'est pas dans le contexte, dis-le.
4. NE FAIS PAS RÉFÉRENCE aux sources du contexte dans ta réponse. Évite les formulations comme :
   - "Selon le document X...", "D'après l'article Y...", "Le fichier Z indique..."
   - "[Source: ...]", "Document 1", "Document 2", noms de fichiers (.html, .pdf)
   Ces références sont internes et n'ont pas de sens pour l'utilisateur.
5. Réponds directement à la question de manière naturelle, comme si tu avais cette connaissance.

RÉPONSE :
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Créer la chaîne en utilisant LCEL (LangChain Expression Language)
        def format_docs(docs):
            # Filtrer et classer les documents pour prioriser la qualité sur la quantité
            if not docs:
                return "Aucun contexte pertinent trouvé."
            
            # Si nous avons beaucoup de docs, préférer les plus courts et plus ciblés en premier
            # pour éviter de submerger le contexte avec du bruit
            formatted_parts = []
            for i, doc in enumerate(docs[:self.top_k], 1):  # Limiter à top_k
                content = doc.page_content.strip()
                # Extraire le titre du document depuis les métadonnées ou le nom de fichier
                source = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else ''
                title = doc.metadata.get('title', '') if hasattr(doc, 'metadata') else ''
                
                # Utiliser le titre si disponible, sinon extraire du chemin source
                if title:
                    doc_label = title
                elif source:
                    # Extraire le nom de fichier et le nettoyer
                    import os
                    filename = os.path.basename(source)
                    # Retirer l'extension et les préfixes numériques
                    doc_label = filename.replace('.html', '').replace('_', ' ')
                    # Retirer le préfixe numérique (ex: "1179 ")
                    if doc_label and doc_label.split(' ')[0].isdigit():
                        doc_label = ' '.join(doc_label.split(' ')[1:])
                else:
                    doc_label = "Documentation"
                
                # Ne pas inclure de référence numérique, juste le contenu
                formatted_parts.append(f"[Source: {doc_label}]\n{content}")
            
            return "\n\n---\n\n".join(formatted_parts)
        
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def _create_retriever(self, vectorstore):
        """Créer un récupérateur basé sur la configuration."""
        
        if self.use_hybrid:
            # Mode hybride : combiner vectoriel + BM25
            return self._create_hybrid_retriever(vectorstore)
        
        elif self.retrieval_mode == "similarity_score_threshold":
            # Récupération basée sur le seuil
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k,
                    "score_threshold": self.score_threshold
                }
            )
        
        else:
            # Recherche de similarité standard
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
    
    def _create_hybrid_retriever(self, vectorstore):
        """Créer un récupérateur hybride combinant recherche vectorielle et BM25."""
        
        # Créer le récupérateur vectoriel
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Charger les documents depuis le vectorstore pour BM25
        # Nous devons obtenir tous les documents de Chroma
        if self.documents is None:
            # Obtenir tous les documents de la collection
            all_docs = vectorstore.get()
            if all_docs and 'documents' in all_docs and all_docs['documents']:
                from langchain_core.documents import Document
                self.documents = [
                    Document(page_content=doc, metadata=meta if meta else {})
                    for doc, meta in zip(
                        all_docs['documents'],
                        all_docs['metadatas'] if 'metadatas' in all_docs else [{}] * len(all_docs['documents'])
                    )
                ]
            else:
                # Repli : pas de documents, retourner seulement le récupérateur vectoriel
                print("Avertissement : Aucun document trouvé pour BM25, utilisation du récupérateur vectoriel uniquement")
                return vector_retriever
        
        # Créer le récupérateur BM25
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.top_k
        
        # Combiner les deux récupérateurs
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=self.hybrid_weights
        )
        
        return ensemble_retriever

    def estimate_dynamic_topk(self, question):
        """
        Estime le top_k optimal en fonction de la complexité de la question.
        """
        # Mots-clés indiquant une recherche large ou procédurale
        complexity_keywords = [
            "comment", "procédure", "étapes", "guide", "tuto", 
            "explication", "fonctionnement", "différence", "liste",
            "tous", "toutes", "quels", "quelles"
        ]
        
        # Mots-clés indiquant une recherche précise
        precision_keywords = [
            "code", "format", "téléphone", "mail", "contact", 
            "adresse", "date", "prix", "montant"
        ]
        
        question_lower = question.lower()
        
        # Score de base
        score = 0
        
        # Facteur longueur (questions longues = souvent plus de contexte nécessaire)
        words = question.split()
        if len(words) > 10:
            score += 2
        elif len(words) > 5:
            score += 1
            
        # Facteur mots-clés complexité
        if any(kw in question_lower for kw in complexity_keywords):
            score += 3
            
        # Facteur connecteurs logiques (indique plusieurs points à couvrir)
        connectors = [" et ", " ou ", " puis ", " ensuite ", " avec "]
        if any(c in question_lower for c in connectors):
            score += 2
            
        # Facteur précision (réduit le besoin de documents)
        if any(kw in question_lower for kw in precision_keywords):
            score -= 2
            
        # Calcul du k final (borné entre 3 et 15)
        base_k = 5
        dynamic_k = max(3, min(15, base_k + score))
        
        return dynamic_k

    def _update_retriever_topk(self, k):
        """Met à jour le k du retriever existant."""
        if not self.retriever:
            return
            
        if isinstance(self.retriever, EnsembleRetriever):
            for retriever in self.retriever.retrievers:
                if hasattr(retriever, "search_kwargs"):
                    retriever.search_kwargs["k"] = k
                elif hasattr(retriever, "k"):
                    retriever.k = k
        else:
            if hasattr(self.retriever, "search_kwargs"):
                self.retriever.search_kwargs["k"] = k

    def ask(self, question, return_sources=False, dynamic_k=True):
        """
        Génère une réponse à partir d'une question.
        
        Args:
            question: La question à poser
            return_sources: Si True, retourne aussi les sources et métadonnées
            dynamic_k: Si True, ajuste automatiquement le top_k selon la complexité
        
        Returns:
            Si return_sources=False: str (la réponse)
            Si return_sources=True: dict avec 'answer', 'sources', 'metadata'
        """
        if not self.qa_chain:
            self.setup_chain()
            
        # Ajustement dynamique du top_k
        current_k = self.top_k
        if dynamic_k:
            estimated_k = self.estimate_dynamic_topk(question)
            # Mettre à jour le retriever
            self._update_retriever_topk(estimated_k)
            current_k = estimated_k
            print(f"Dynamic top_k: {estimated_k} (base: {self.top_k})")
        
        # Générer la réponse
        response = self.qa_chain.invoke(question)
        
        if not return_sources:
            # Restaurer le k original si besoin (bonnes pratiques)
            if dynamic_k:
                self._update_retriever_topk(self.top_k)
            return response
        
        # Récupérer les documents sources utilisés
        try:
            docs = self.retriever.invoke(question)
            sources = []
            
            for doc in docs[:current_k]:
                source_info = {
                    'source': doc.metadata.get('source', 'Unknown'),
                    'filename': doc.metadata.get('filename', 'Unknown'),
                    'category': doc.metadata.get('category', 'Unknown'),
                    'post_id': doc.metadata.get('post_id', ''),
                    'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            # Restaurer le k original
            if dynamic_k:
                self._update_retriever_topk(self.top_k)
            
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
        except Exception as e:
            # En cas d'erreur, retourner quand même la réponse
            print(f"Warning: Could not retrieve sources: {e}")
            return {
                'answer': response,
                'sources': [],
                'metadata': {'error': str(e)}
            }
