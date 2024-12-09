from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.tool import ToolType, ToolOutput
from ..resources.tool import Tool
from typing import List, Optional
import json


class ToolsManager(SyncAPIResource):

    _client: RelevanceAI

    def list_tools(
        self,
        max_results: Optional[int] = 100,
    ) -> List[Tool]:
        path = "studios/list"
        params = {
            "filters": json.dumps(
                [
                    {
                        "filter_type": "exact_match",
                        "field": "project",
                        "condition_value": self._client.project,
                        "condition": "==",
                    }
                ]
            ),
            "page_size": max_results,
        }
        response = self._client.get(path, params=params)
        tools = [Tool(client=self._client, **item) for item in response.json().get("results", [])]
        return tools
    
    def retrieve_tool(self, tool_id: str) -> Tool:
        path = f"studios/{tool_id}/get"
        response = self._get(path)
        return Tool(client=self._client, **response.json()["studio"])