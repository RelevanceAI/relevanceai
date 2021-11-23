import os
import requests

PATH = os.path.dirname(os.path.abspath(__file__))
API_PATH = os.path.join(PATH, "relevanceai/api")
MODULES = os.listdir(API_PATH)
MODULES.remove('__pycache__')
ENDPOINTS = list(requests.get("https://gateway-api-aueast.relevance.ai/latest/openapi.json").json()['paths'].keys())

endpoint_in_sdk = []
endpoint_not_in_sdk = ENDPOINTS.copy()
for module in MODULES:
    with open(os.path.join(API_PATH, module)) as f:
        lines = f.readlines()
        for line in lines:
            for endpoint in endpoint_not_in_sdk:
                if endpoint in line:
                    endpoint_in_sdk.append(endpoint)
                    endpoint_not_in_sdk.remove(endpoint)


with open(os.path.join(PATH, "endpoint_in_sdk.txt"), 'w') as f:
    for endpoint in endpoint_in_sdk:
        f.write(f"{endpoint}\n")

with open(os.path.join(PATH, "endpoint_not_in_sdk.txt"), 'w') as f:
    for endpoint in endpoint_not_in_sdk:
        f.write(f"{endpoint}\n")