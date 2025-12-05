import sqlite3
import json
import os
import time
import hashlib

class CacheManager:
    def __init__(self, db_path=".cache/rag_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
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

    def _get_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query, top_k):
        query_hash = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response, sources FROM cache WHERE query_hash = ? AND top_k >= ?", 
                (query_hash, top_k)
            )
            row = cursor.fetchone()
            
            if row:
                print(f"✓ Cache HIT (disk) pour: '{query[:50]}...'")
                return {
                    'answer': row[0],
                    'sources': json.loads(row[1]) if row[1] else []
                }
        return None

    def set(self, query, top_k, result):
        """
        Cache the result dict which contains 'answer' and 'sources'
        """
        query_hash = self._get_hash(query)
        response = result.get('answer', '')
        sources = result.get('sources', [])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (query_hash, query, response, sources, timestamp, top_k) VALUES (?, ?, ?, ?, ?, ?)",
                (query_hash, query, response, json.dumps(sources), time.time(), top_k)
            )
