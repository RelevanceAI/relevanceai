from __future__ import annotations
import time
from typing import TYPE_CHECKING, Type, TypeVar, Dict, Any, Optional, Union
import httpx
if TYPE_CHECKING:
    from ._client import RelevanceAI

ResponseT = TypeVar('ResponseT')

class SyncAPIResource:
    _client: RelevanceAI
    
    def __init__(self, client: RelevanceAI) -> None:
        self._client = client

    def _get(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None
    ) -> ResponseT:
        response = self._client.get(path, json=body, params=params)
        return self._cast_response(response, cast_to)

    def _post(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.post(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _patch(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.patch(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _put(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.put(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _delete(
        self, 
        path: str, 
        cast_to: Type[ResponseT] = None, 
        body: Optional[Dict[str, Any]] = None, 
        params: Optional[Dict[str, Any]] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> ResponseT:
        options = options or {}
        response = self._client.delete(path, json=body, params=params, **options)
        return self._cast_response(response, cast_to)

    def _cast_response(
        self, 
        response: httpx.Response, 
        cast_to: Type[ResponseT] = None
    ) -> Union[ResponseT, dict, httpx.Response]:
        if cast_to:
            return response.json() if cast_to == dict else cast_to(**response.json())
        return response

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)