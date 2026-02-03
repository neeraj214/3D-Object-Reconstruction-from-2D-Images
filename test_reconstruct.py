
import requests

url = "http://127.0.0.1:8000/api/reconstruct"
files = {'file': open('category_images/pix3d/bed/0265_original.png', 'rb')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(response.text[:500])  # Print first 500 chars of response
except Exception as e:
    print(f"Error: {e}")
