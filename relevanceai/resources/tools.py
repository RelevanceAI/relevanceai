from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.tool import Tool, ToolOutput
from typing import List, Optional
import json 

class Tools(SyncAPIResource):

    _client: RelevanceAI
    
    def list_tools(
        self,
        max_results: Optional[int] = 100,
    ) -> List[Tool]:
        path = "studios/list"
        params = {
            "filters": json.dumps([{
                "filter_type": "exact_match",
                "field": "project",
                "condition_value": self._client.project,
                "condition": "=="
            }]),
            "page_size": max_results
        }
        response = self._client.get(path, params=params)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        return tools

    def retrieve_tool(self, tool_id: str) -> Tool:
        path = f"studios/{tool_id}/get"
        response = self._get(path)
        return Tool(**response.json()["studio"])
    
    def trigger_tool(
        self,
        tool_id: str,
        params: dict | None = None,
    ): 
        path = f"studios/{tool_id}/trigger_limited"
        body = {
            "params": params,
            "project": self._client.project
        }
        response = self._post(path=path, body=body)
        return ToolOutput(**response.json())
    
    def _get_params_as_json_string(
        self,
        tool_id: str,
    ) -> str:
        response = self._get(f"studios/{tool_id}/get")
        params_schema = response.json()["studio"]["params_schema"]["properties"]
        return json.dumps(params_schema, indent=4)
        
    def _get_steps_as_json_string(
        self,
        tool_id: str
    ) -> str:
        response = self._get(f"studios/{tool_id}/get")
        transformations = response.json()["studio"]["transformations"]
        steps = transformations["steps"]
        return json.dumps(steps, indent=4)

    def delete_tool(self, tool_id: str) -> bool:
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        response = self._post(path, body=body)
        return response.status_code == 200

    def list_agent_tools(
        self,
        agent_id: str,
    ) -> List[Tool]:
        path = "agents/tools/list"
        body = {"agent_ids": [agent_id]}
        response = self._client.post(path, body=body)
        tools = [Tool(**item) for item in response.json().get("results", [])]
        return tools