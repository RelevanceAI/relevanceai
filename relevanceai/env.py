import requests
from relevanceai._request import handle_response
from relevanceai import config

def set_key(key:str, value:str):
    url = f"{config.auth.url}/latest"
    response = requests.post(
        f"{url}/projects/keys/set",
        headers=config.auth.headers,
        json={
            "key": key,
            "value": value,
        }
    )
    res = handle_response(response)
    return res

def list_keys():
    url = f"{config.auth.url}/latest"
    response = requests.get(
        f"{url}/projects/keys/list",
        headers=config.auth.headers,
    )
    res = handle_response(response)
    return res

def delete_key(key:str):
    url = f"{config.auth.url}/latest"
    response = requests.post(
        f"{url}/projects/keys/delete",
        headers=config.auth.headers,
        json={
            "key": key,
        }
    )
    res = handle_response(response)
    return res