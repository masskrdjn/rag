from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="/home/ragapp/rag-system/chroma_db", embedding_function=embeddings)
    
    print("Querying for 'congés payés'...")
    results = vectorstore.similarity_search("congés payés", k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print("-" * 20)

except Exception as e:
    print(f"Error: {e}")
