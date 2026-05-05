"""
Cache disque (SQLite) pour les réponses RAG : clé = (hash de la question, top_k).
Seules les réponses avec une confiance suffisante sont mises en cache (cf. rag_pipeline.py).
"""

import hashlib
import json
import os
import sqlite3
import time
from typing import Dict, Optional


class CacheManager:
    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache", "rag_cache.db"
    )

    def __init__(self, db_path: str = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    response TEXT,
                    sources TEXT,
                    timestamp REAL,
                    top_k INTEGER
                )
            """)

    @staticmethod
    def _get_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str, top_k: int) -> Optional[Dict]:
        query_hash = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, sources FROM cache WHERE query_hash = ? AND top_k >= ?",
                (query_hash, top_k),
            )
            row = cursor.fetchone()
            if row:
                print(f"Cache HIT pour : '{query[:50]}...'")
                return {
                    "answer": row[0],
                    "sources": json.loads(row[1]) if row[1] else [],
                }
        return None

    def set(self, query: str, top_k: int, result: Dict) -> None:
        query_hash = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(query_hash, query, response, sources, timestamp, top_k) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    query_hash,
                    query,
                    result.get("answer", ""),
                    json.dumps(result.get("sources", [])),
                    time.time(),
                    top_k,
                ),
            )

    def clear(self) -> int:
        """Vide entièrement le cache et retourne le nombre d'entrées supprimées."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            conn.execute("DELETE FROM cache")
        return count
