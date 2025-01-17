from __future__ import annotations
import httpx 
from httpx import Timeout, URL, Headers, Response
from httpx._types import ProxiesTypes

class SyncAPIClient:
    
    _client: httpx.Client

    def __init__(
        self,
        *,
        base_url: str | URL,
        headers: Headers | None = None,
        timeout: float | Timeout | None = None,
        proxies: ProxiesTypes | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout, proxies=proxies)

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return self._client.request(method, url, **kwargs)
    
    def get(self, path: str, **kwargs) -> Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return self.request("PATCH", path, **kwargs)

    def put(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> Response:
        return self.request("DELETE", path, **kwargs)

    def close(self) -> None:
        self._client.close()

class AsyncAPIClient:
    _client: httpx.AsyncClient

    def __init__(
        self,
        *,
        base_url: str | URL,
        headers: Headers | None = None,
        timeout: float | Timeout | None = None,
        proxies: ProxiesTypes | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, 
            headers=headers, 
            timeout=timeout, 
            proxies=proxies
        )

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return await self._client.request(method, url, **kwargs)
    
    async def get(self, path: str, **kwargs) -> Response:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return await self.request("POST", path, **kwargs)

    async def patch(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return await self.request("PATCH", path, **kwargs)

    async def put(self, path: str, body: dict = None, **kwargs) -> Response:
        if body is not None:
            kwargs['json'] = body
        return await self.request("PUT", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> Response:
        return await self.request("DELETE", path, **kwargs)

    async def close(self) -> None:
        await self._client.aclose()