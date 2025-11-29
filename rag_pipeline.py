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
        self.persist_directory = "chroma_db"
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
        retriever = self._create_retriever(vectorstore)
            
        # Initialiser le LLM
        llm = ChatOllama(model=self.model_name)
        
        # Créer le modèle de prompt
        template = """Tu es un assistant interne qui aide les employés à appliquer les procédures de l'entreprise.

RÈGLES STRICTES - TU DOIS LES SUIVRE À LA LETTRE :
1. Réponds TOUJOURS en français, de façon concise et professionnelle.
2. Utilise UNIQUEMENT et EXCLUSIVEMENT les informations exactes du contexte ci-dessous.
3. INTERDICTION ABSOLUE d'inventer, déduire, ou compléter des informations qui ne sont pas EXPLICITEMENT dans le contexte.
4. Si le contexte dit "consulter X" ou "voir Y", réponds EXACTEMENT cela. N'invente PAS d'étapes supplémentaires.
5. Si le contexte mentionne une source (Whaller, Gravity, email, lien), cite-la TEXTUELLEMENT telle qu'elle apparaît.
6. Si une information n'est PAS dans le contexte, réponds : "Je n'ai pas cette information dans ma base de connaissances."
7. Ne JAMAIS inventer de liens, URLs, adresses email, noms d'outils, formulaires ou procédures.
8. Ne JAMAIS mélanger des informations de documents différents.
9. Si le Document 1 répond à la question, utilise UNIQUEMENT celui-ci et ignore les autres.

FORMAT DE RÉPONSE :
- Répète TEXTUELLEMENT ce que dit le contexte
- Cite la source si elle est mentionnée
- Reste concis et factuel

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""
        
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
                source = doc.metadata.get('source', 'Source inconnue') if hasattr(doc, 'metadata') else 'Source inconnue'
                
                # Ajouter une référence de source pour la traçabilité
                formatted_parts.append(f"[Document {i} - {source}]\n{content}")
            
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

    def ask(self, question):
        if not self.qa_chain:
            self.setup_chain()
        # La chaîne retourne une chaîne directement
        response = self.qa_chain.invoke(question)
        return response
