#!/usr/bin/env python3
"""Test direct de l'API : /health puis une question simple."""

import json
import sys

import requests

API_BASE = "http://localhost:8000"


def main() -> None:
    print("=" * 60)
    print("TEST API RAG")
    print("=" * 60)

    print("\n1. /health")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=30)
        print(f"   Status : {r.status_code} | {r.json()}")
    except Exception as e:
        print(f"   ERREUR : {e}")
        sys.exit(1)

    print("\n2. /ask")
    question = "quelles sont les regles des conges"
    print(f"   Question : '{question}'")
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=120,
        )
        r.raise_for_status()
        print(f"   Status : {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"   ERREUR : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
