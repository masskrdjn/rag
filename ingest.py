import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def ingest_documents():
    print(f"Loading documents from {DATA_PATH}...")
    # Load documents
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Save to Chroma
    print("Saving to ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Create/Update vector store
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Ingestion complete! Data saved to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_documents()
