import requests


url = "http://localhost:5000"

response = requests.get(f"{url}/ee")
print(response.status_code)
print(response.text)

data = response.json()
print(data)