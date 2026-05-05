"""
Cache disque (SQLite) pour les réponses RAG : clé = (hash de la question, top_k).
Seules les réponses avec une confiance suffisante sont mises en cache (cf. rag_pipeline.py).
"""

import hashlib
import json
import os
import re
import sqlite3
import time
import unicodedata
from typing import Dict, Optional


class CacheManager:
    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache", "rag_cache.db"
    )

    def __init__(self, db_path: str = None, ttl_seconds: int = None, namespace: str = ""):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
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
                    metadata TEXT,
                    timestamp REAL,
                    top_k INTEGER
                )
            """)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(cache)").fetchall()
            }
            if "metadata" not in columns:
                conn.execute("ALTER TABLE cache ADD COLUMN metadata TEXT")

    @staticmethod
    def _normalize_query(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.strip().lower())
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _get_hash(payload: Dict) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _query_hash(self, query: str, top_k: int, namespace: str = None) -> str:
        payload = {
            "query": self._normalize_query(query),
            "top_k": top_k,
            "namespace": namespace if namespace is not None else self.namespace,
        }
        return self._get_hash(payload)

    def get(self, query: str, top_k: int, namespace: str = None) -> Optional[Dict]:
        query_hash = self._query_hash(query, top_k, namespace)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, sources, metadata, timestamp FROM cache "
                "WHERE query_hash = ? AND top_k >= ?",
                (query_hash, top_k),
            )
            row = cursor.fetchone()
            if row:
                timestamp = row[3] or 0
                if self.ttl_seconds and time.time() - timestamp > self.ttl_seconds:
                    conn.execute("DELETE FROM cache WHERE query_hash = ?", (query_hash,))
                    return None
                print(f"Cache HIT pour : '{query[:50]}...'")
                metadata = json.loads(row[2]) if row[2] else {}
                metadata["cache_hit"] = True
                return {
                    "answer": row[0],
                    "sources": json.loads(row[1]) if row[1] else [],
                    "metadata": metadata,
                }
        return None

    def set(self, query: str, top_k: int, result: Dict, namespace: str = None) -> None:
        query_hash = self._query_hash(query, top_k, namespace)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(query_hash, query, response, sources, metadata, timestamp, top_k) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    query_hash,
                    self._normalize_query(query),
                    result.get("answer", ""),
                    json.dumps(result.get("sources", [])),
                    json.dumps(result.get("metadata", {}), default=str),
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
