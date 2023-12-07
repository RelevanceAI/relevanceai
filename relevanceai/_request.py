import requests

from relevanceai.types import JSONObject


def _handle_response(response: requests.Request) -> JSONObject:
    try:
        return response.json()
    except:
        return response.text
