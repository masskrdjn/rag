"""
Benchmark leger des embeddings Ollama pour le RAG.

Important: chaque modele d'embedding doit avoir sa propre base Chroma re-ingeree.
Exemple:
  RAG_EMBEDDING_MODEL=nomic-embed-text RAG_CHROMA_DB_PATH=./chroma_nomic python ingest_html_adaptive.py
  RAG_EMBEDDING_MODEL=bge-m3 RAG_CHROMA_DB_PATH=./chroma_bge_m3 python ingest_html_adaptive.py
  python benchmark_embeddings.py --embedding nomic-embed-text --chroma ./chroma_nomic --embedding bge-m3 --chroma ./chroma_bge_m3
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_questions(path: Path, limit: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        questions = [
            item.get("question", item) if isinstance(item, dict) else item
            for item in data
        ]
    elif isinstance(data, dict):
        questions = [
            item.get("question", item) if isinstance(item, dict) else item
            for item in data.get("questions", [])
        ]
    else:
        questions = []
    return [str(q) for q in questions if str(q).strip()][:limit]


def run_worker(args) -> int:
    from rag_pipeline import SimpleRAG

    questions = load_questions(Path(args.questions), args.limit)
    rag = SimpleRAG()
    rag.setup_chain()

    results = []
    for question in questions:
        result = rag.ask(question, return_sources=True, dynamic_k=True)
        metadata = result.get("metadata", {})
        results.append({
            "question": question,
            "num_sources": len(result.get("sources", [])),
            "confidence": metadata.get("confidence"),
            "execution_time": metadata.get("execution_time"),
            "top_k": metadata.get("top_k"),
            "action": metadata.get("action"),
        })

    output = {
        "embedding": os.environ.get("RAG_EMBEDDING_MODEL"),
        "chroma": os.environ.get("RAG_CHROMA_DB_PATH"),
        "results": results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", action="append", required=True)
    parser.add_argument("--chroma", action="append", required=True)
    parser.add_argument("--questions", default="test_questions.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    if len(args.embedding) != len(args.chroma):
        parser.error("--embedding et --chroma doivent etre fournis par paires")

    for embedding, chroma in zip(args.embedding, args.chroma):
        env = os.environ.copy()
        env["RAG_EMBEDDING_MODEL"] = embedding
        env["RAG_CHROMA_DB_PATH"] = chroma
        env["RAG_CORPUS_BUILD_ID"] = f"bench-{embedding}-{Path(chroma).name}"
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--embedding",
            embedding,
            "--chroma",
            chroma,
            "--questions",
            args.questions,
            "--limit",
            str(args.limit),
        ]
        print(f"\n=== Benchmark {embedding} ({chroma}) ===")
        subprocess.run(cmd, env=env, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
