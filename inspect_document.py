from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys

# Force stdout to use utf-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="/home/ragapp/rag-system/chroma_db", embedding_function=embeddings)
    
    # Search by metadata
    print("Searching for document '1068_Conges_payes_et_absences.html'...")
    
    # Chroma doesn't have a direct "get by metadata" in LangChain interface easily, 
    # but we can filter. Or just search for the specific title which should be unique now.
    results = vectorstore.similarity_search("Congés payés et absences", k=1, filter={"source": "/home/ragapp/rag-system/data/1068_Conges_payes_et_absences.html"})
    
    if results:
        doc = results[0]
        print(f"\nFound Document:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content:\n{doc.page_content}")
    else:
        print("Document not found with that source metadata.")
        
        # Try searching without filter just in case path is different
        print("\nTrying loose search...")
        results = vectorstore.similarity_search("Congés payés et absences", k=5)
        for doc in results:
            if "Congés" in doc.page_content:
                print(f"Found candidate: {doc.metadata.get('source')}")
                print(f"Content preview: {doc.page_content[:100]}...")

except Exception as e:
    print(f"Error: {e}")
