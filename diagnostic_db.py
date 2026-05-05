#!/usr/bin/env python3
"""Diagnostic de la base ChromaDB (taille, recherche test, métadonnées)."""

import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DB_PATH, EMBEDDING_MODEL


def main() -> None:
    print("=" * 60)
    print("DIAGNOSTIC BASE CHROMADB")
    print("=" * 60)
    print(f"Chemin : {CHROMA_DB_PATH}")
    print(f"Embeddings : {EMBEDDING_MODEL}")

    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
        )

        count = vectorstore._collection.count()
        print(f"\nTotal : {count} chunks dans la base")
        if count == 0:
            print("LA BASE EST VIDE !")
            sys.exit(1)

        for query in ("congés payés", "regles conges absences"):
            print(f"\nRecherche : '{query}'")
            results = vectorstore.similarity_search_with_score(query, k=3)
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", "N/A")
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"  [{i}] score={score:.4f} | {source}")
                print(f"      {preview}...")

        print("\nDiagnostic terminé.")
    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
