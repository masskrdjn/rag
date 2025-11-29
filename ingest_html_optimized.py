import os
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = "/home/ragapp/rag-system/data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def extract_images_from_html(html_content, source_file):
    """
    Extrait les URLs d'images du HTML et les ajoute au texte
    pour qu'elles soient indexées dans la base vectorielle.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extraire le titre
    title = soup.title.string if soup.title else ""
    
    # Supprimer les scripts et styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text(separator='\n')
    
    # Enrichir le contexte pour les documents courts ou de redirection
    if title:
        text = f"Titre du document : {title}\n\n{text}"
        
    if "Whaller" in text and "Consulter" in text:
        text += "\n\nNOTE IMPORTANTE : Pour les détails complets et les règles officielles, il est impératif de consulter la sphère Whaller indiquée ci-dessus."
    
    # Extraire les images
    images = soup.find_all('img')
    
    if images:
        text += "\n\n=== IMAGES RÉFÉRENCÉES ===\n"
        for idx, img in enumerate(images, 1):
            src = img.get('src', '')
            alt = img.get('alt', 'Sans description')
            if src:
                text += f"\nImage {idx}: {alt}\nURL: {src}\n"
    
    return text

def load_html_documents_with_images(data_path):
    """
    Charge tous les fichiers HTML du répertoire data et extrait
    le texte + les URLs d'images.
    """
    documents = []
    data_dir = Path(data_path)
    
    # Chercher récursivement tous les fichiers HTML
    html_files = list(data_dir.rglob("*.html")) + list(data_dir.rglob("*.htm"))
    
    print(f"Found {len(html_files)} HTML file(s)")
    
    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extraire texte + images
            text = extract_images_from_html(html_content, str(html_file))
            
            # Créer un Document LangChain avec metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(html_file),
                    "filename": html_file.name,
                    "type": "html"
                }
            )
            documents.append(doc)
            print(f"✓ Loaded: {html_file.name}")
            
        except Exception as e:
            print(f"✗ Error loading {html_file.name}: {e}")
    
    return documents

def ingest_html_documents_optimized():
    """
    Ingestion optimisée avec:
    - Extraction des URLs d'images
    - Séparateurs personnalisés pour structure WordPress
    - Paramètres de chunking adaptés aux procédures
    """
    print(f"Loading HTML documents from {DATA_PATH}...")
    print("=" * 60)
    
    # Charger les documents avec extraction d'images
    documents = load_html_documents_with_images(DATA_PATH)
    
    if len(documents) == 0:
        print("⚠ No HTML documents found. Check your data path.")
        return
    
    print(f"\n✓ Loaded {len(documents)} HTML document(s)")
    
    # Calculer la taille totale du texte
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"Total characters: {total_chars:,}")
    
    # Configuration optimale du text splitter
    # Séparateurs adaptés au contenu WordPress/procédures
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Chunks plus petits = meilleure précision
        chunk_overlap=100,     # 12.5% overlap = bon équilibre
        separators=[
            "\n\n",            # Paragraphes
            "\n- ",            # Listes à puces
            "\n* ",            # Listes alternatives
            "\n# ",            # Titres markdown H1
            "\n## ",           # Titres markdown H2
            "\n### ",          # Titres markdown H3
            "\n",              # Lignes simples
            " ",               # Mots
            "",                # Caractères
        ],
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    print(f"Average chunk size: {total_chars // len(chunks)} characters")
    
    # Afficher un échantillon
    if chunks:
        print("\n" + "=" * 60)
        print("SAMPLE CHUNK (first one):")
        print("=" * 60)
        sample = chunks[0]
        print(f"Source: {sample.metadata.get('source', 'N/A')}")
        print(f"Content preview (first 300 chars):")
        print(sample.page_content[:300] + "...")
        print("=" * 60)
    
    # Sauvegarder dans ChromaDB
    print("\nSaving to ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Créer/mettre à jour le vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Force persistence if available
    try:
        if hasattr(vectorstore, 'persist'):
            vectorstore.persist()
            print("✓ Explicitly persisted to disk")
    except Exception as e:
        print(f"⚠ Could not explicitly persist: {e}")
    
    print(f"\n{'=' * 60}")
    print("✓ INGESTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\n📊 Summary:")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Average chunk size: {total_chars // len(chunks)} chars")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - Database location: {CHROMA_PATH}")
    print(f"\n💡 Tips:")
    print(f"  - Images URLs are now indexed and searchable")
    print(f"  - Chunks are optimized for procedural content")
    print(f"  - Don't forget to restart your RAG server!")

if __name__ == "__main__":
    ingest_html_documents_optimized()
