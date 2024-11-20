from __future__ import annotations

from .._resource import SyncAPIResource
from .._client import RelevanceAI
from typing import List
from ..types.knowledge import KnowledgeSet, KnowledgeRow

class Knowledge(SyncAPIResource):

    _client: RelevanceAI

    def list_knowledge(
        self, 
    ) -> List[KnowledgeSet]:
        path = "knowledge/sets/list"
        body = {
            "filters": [],
            "sort": [{"update_date":"desc"}]
        }
        response = self._post(path, body=body)
        return [KnowledgeSet(**item) for item in response.json().get("results", [])]
    
    def retrieve_knowledge(
        self, 
        knowledge_set: str, 
        max_results: int = 5
    ) -> List[KnowledgeRow]:
        path = "knowledge/list"
        body = {
            "knowledge_set": knowledge_set,
            "page_size": max_results,
            "sort": [{"insert_date_": "asc"}]
        }
        response = self._post(path, body=body)
        return [KnowledgeRow(**item) for item in response.json().get("results", [])]
        
    def delete_knowledge(
        self,
        knowledge_set: str, 
    ) -> bool:
        path = "knowledge/sets/delete"
        body = {"knowledge_set": knowledge_set}
        response = self._post(path, body=body)
        return response.status_code == 200

