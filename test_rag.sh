#!/bin/bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Quelle est la hauteur de la Tour Eiffel ?"}'
