import requests

from relevanceai.types import JSONObject


def _handle_response(response: requests.Request) -> JSONObject:
    try:
        return response.json()
    except:
        print(response.status_code)
        return response
