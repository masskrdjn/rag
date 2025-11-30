import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    print("--- Direct ChromaDB Inspection ---")
    client = chromadb.PersistentClient(path="/home/rag/chroma_db")
    print(f"Collections: {client.list_collections()}")
    
    for col in client.list_collections():
        print(f"Collection '{col.name}' count: {col.count()}")
        if col.count() > 0:
            print(f"Sample doc: {col.peek(limit=1)}")

    print("\n--- LangChain Inspection ---")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="/home/rag/chroma_db", embedding_function=embeddings)
    
    print(f"Total docs in collection (LangChain): {len(vectorstore.get()['ids'])}")
    
    query = "quelles sont les règles des congés ?"
    print(f"Query: '{query}'")
    
    # Get results with scores (distance)
    # Lower distance = better match
    results_with_score = vectorstore.similarity_search_with_score(query, k=5)
    
    print(f"\nFound {len(results_with_score)} results:")
    for i, (doc, score) in enumerate(results_with_score):
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200].replace(chr(10), ' ')}...")
        
except Exception as e:
    print(f"Error: {e}")
