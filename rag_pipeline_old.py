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
                 top_k=3,
                 score_threshold=0.7,
                 use_hybrid=False,
                 hybrid_weights=None):
        """
        Initialize RAG pipeline with configurable retrieval strategies.
        
        Args:
            retrieval_mode (str): "similarity" or "similarity_score_threshold"
            top_k (int): Number of documents to retrieve (3-8 recommended)
            score_threshold (float): Minimum similarity score (0.5-0.8, only for threshold mode)
            use_hybrid (bool): Whether to use hybrid BM25 + vector search
            hybrid_weights (list): Weights for [vector, bm25] (default [0.5, 0.5])
        """
        self.persist_directory = "chroma_db"
        self.model_name = "llama3.2"
        self.embedding_model = "nomic-embed-text"
        self.qa_chain = None
        
        # Retrieval configuration
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_hybrid = use_hybrid
        self.hybrid_weights = hybrid_weights if hybrid_weights else [0.5, 0.5]
        
        # Cache for documents (needed for BM25)
        self.documents = None

    def setup_chain(self):
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Initialize vector store
        vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # Configure retriever based on mode
        retriever = self._create_retriever(vectorstore)
            
        # Initialize LLM
        llm = ChatOllama(model=self.model_name)
        
        # Create prompt template
        template = """Tu es un assistant interne qui aide les employés à appliquer les procédures de l’entreprise.

RÈGLES :
- Réponds TOUJOURS en français sauf si le contraire est explicitement demandé, de façon concise et professionnelle.
- Appuie-toi UNIQUEMENT sur le contexte fourni. Si une information n’est pas dans le contexte, dis que tu ne l’as pas.
- Si plusieurs procédures sont possibles, liste-les clairement.
- Si le contexte indique une source (nom de procédure, lien, outil interne comme Whaller, Gravity, site web, etc.), mentionne-la explicitement dans la réponse.
- Si le contexte est insuffisant ou ambigu, pose des questions de clarification à l’utilisateur.
- Ne jamais inventer de lien ou d’URL.
- Ne jamais répondre sur un sujet RH/juridique sans contexte explicite dans la base.

FORMAT DE RÉPONSE :
1. Réponse courte (2–3 phrases maximum) qui va droit au but.
2. Étapes détaillées numérotées si la réponse décrit une procédure.
3. Références : liste les titres d’articles, dates, et liens issus du contexte utilisés pour répondre.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            # Filter and rank documents to prioritize quality over quantity
            if not docs:
                return "Aucun contexte pertinent trouvé."
            
            # If we have many docs, prefer shorter, more focused ones first
            # to avoid overwhelming the context with noise
            formatted_parts = []
            for i, doc in enumerate(docs[:self.top_k], 1):  # Limit to top_k
                content = doc.page_content.strip()
                source = doc.metadata.get('source', 'Source inconnue') if hasattr(doc, 'metadata') else 'Source inconnue'
                
                # Add source reference for traceability
                formatted_parts.append(f"[Document {i} - {source}]\n{content}")
            
            return "\n\n---\n\n".join(formatted_parts)
        
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def _create_retriever(self, vectorstore):
        """Create retriever based on configuration."""
        
        if self.use_hybrid:
            # Hybrid mode: combine vector + BM25
            return self._create_hybrid_retriever(vectorstore)
        
        elif self.retrieval_mode == "similarity_score_threshold":
            # Threshold-based retrieval
            return vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.top_k,
                    "score_threshold": self.score_threshold
                }
            )
        
        else:
            # Standard similarity search
            return vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
    
    def _create_hybrid_retriever(self, vectorstore):
        """Create hybrid retriever combining vector search and BM25."""
        
        # Create vector retriever
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Load documents from vectorstore for BM25
        # We need to get all documents from Chroma
        if self.documents is None:
            # Get all documents from the collection
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
                # Fallback: no documents, return only vector retriever
                print("Warning: No documents found for BM25, using vector retriever only")
                return vector_retriever
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = self.top_k
        
        # Combine both retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=self.hybrid_weights
        )
        
        return ensemble_retriever

    def ask(self, question):
        if not self.qa_chain:
            self.setup_chain()
        # Chain returns a string directly
        response = self.qa_chain.invoke(question)
        return response


