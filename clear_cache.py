#!/usr/bin/env python3
"""Vide le cache du RAG (table SQLite)."""

from cache_manager import CacheManager


def main() -> None:
    cache = CacheManager()
    deleted = cache.clear()
    if deleted == 0:
        print("Cache déjà vide.")
    else:
        print(f"{deleted} entrée(s) supprimée(s) du cache : {cache.db_path}")


if __name__ == "__main__":
    main()
