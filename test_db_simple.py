from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embeddings)

print(f'Total documents: {vectorstore._collection.count()}')

results = vectorstore.similarity_search('congés payés règles', k=3)
print(f'Results found: {len(results)}')

for i, doc in enumerate(results, 1):
    print(f"\n--- Doc {i} ---")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Content: {doc.page_content[:200]}...")
