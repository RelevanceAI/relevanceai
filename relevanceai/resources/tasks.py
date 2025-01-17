from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..types.knowledge import Metadata
from typing import Optional, List, Union

class Tasks(SyncAPIResource):

    _client: RelevanceAI

    def get_metadata(self, conversation_id: str) -> Union[Metadata, dict]:
        path = f"knowledge/sets/{conversation_id}/get_metadata"
        response = self._get(path).json()
        return Metadata(**response["metadata"]) if response.get("metadata") else {}

    def delete_task(self, conversation_id: str) -> bool:
        path = "knowledge/sets/delete"
        body = {"knowledge_set": [conversation_id]}
        response = self._post(path=path, body=body)
        return response.status_code == 200

class AsyncTasks(AsyncAPIResource):

    _client: AsyncRelevanceAI

    async def get_metadata(self, conversation_id: str) -> Union[Metadata, dict]:
        path = f"knowledge/sets/{conversation_id}/get_metadata"
        response = await self._get(path)
        response_json = response.json()
        return Metadata(**response_json["metadata"]) if response_json.get("metadata") else {}

    async def delete_task(self, conversation_id: str) -> bool:
        path = "knowledge/sets/delete"
        body = {"knowledge_set": [conversation_id]}
        response = await self._post(path=path, body=body)
        return response.status_code == 200
