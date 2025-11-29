import requests
import json

url = "http://localhost:8000/ask"
data = {"question": "quelles sont les règles des congés ?"}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
