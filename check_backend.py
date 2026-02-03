import requests
try:
    print(requests.get('http://127.0.0.1:8000/api/status').text)
except Exception as e:
    print("Error:", e)
