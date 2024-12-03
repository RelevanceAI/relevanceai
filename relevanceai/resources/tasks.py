from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.task import Task, TriggeredTask, ScheduledActionTrigger, TaskView, TaskStep
from ..types.knowledge import Metadata
from typing import Optional, List, Union
import json


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
