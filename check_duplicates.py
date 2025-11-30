from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="/home/rag/chroma_db", embedding_function=embeddings)
    
    print("Checking for duplicates of '1068_Conges_payes_et_absences.html'...")
    
    # Get all docs (this might be heavy if DB is huge, but we have ~87 docs)
    # Chroma client direct access might be better but let's try similarity search with high k
    results = vectorstore.similarity_search("Congés", k=100)
    
    count = 0
    versions = []
    for doc in results:
        if "1068_Conges_payes_et_absences.html" in doc.metadata.get("source", ""):
            count += 1
            preview = doc.page_content[:50].replace('\n', ' ')
            versions.append(preview)
            
    print(f"Found {count} occurrences of the Congés document.")
    for i, v in enumerate(versions):
        print(f"Version {i+1}: {v}...")

except Exception as e:
    print(f"Error: {e}")
