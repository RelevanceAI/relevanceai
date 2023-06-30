import requests
from relevanceai.helpers import _handle_response
from relevanceai import config


def set_key(key: str, value: str):
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com/latest"
    response = requests.post(
        f"{url}/projects/keys/set",
        headers=config.auth.headers,
        json={
            "key": key,
            "value": value,
        },
    )
    return _handle_response(response)


def list_keys():
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com/latest"
    response = requests.get(
        f"{url}/projects/keys/list",
        headers=config.auth.headers,
    )
    return _handle_response(response)


def delete_key(key: str):
    url = f"https://api-{config.auth.region}.stack.tryrelevance.com/latest"
    response = requests.post(
        f"{url}/projects/keys/delete",
        headers=config.auth.headers,
        json={
            "key": key,
        },
    )
    return _handle_response(response)
