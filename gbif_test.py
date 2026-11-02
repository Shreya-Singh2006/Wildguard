import requests
from getpass import getpass

username = input("GBIF Username: ")
password = getpass("GBIF Password: ")

url = "https://api.gbif.org/v1/occurrence/search"

response = requests.get(
    url,
    auth=(username, password),
    params={"limit": 1}
)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("GBIF connection successful!")
else:
    print("Authentication failed:", response.text)