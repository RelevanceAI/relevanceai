from __future__ import annotations

from .._client import RelevanceAI, AsyncRelevanceAI
from .._resource import SyncAPIResource, AsyncAPIResource
from ..types.tool import ToolType, ToolOutput
from ..resources.tool import Tool, AsyncTool
from typing import List, Optional
import json
import uuid


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
    
    def create_tool(
        self, 
        title: str, 
        description: str, 
        public: bool = False,
        params_schema = None,
        output_schema = None,
        transformations = None
    ) -> Tool:
        tool_id = uuid.uuid4()
        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "title": title,
                    "public": public,
                    "project": self._client.project,
                    "description": description,
                    "version": "latest",
                    "params_schema": {
                        "properties": {},
                        "required": [],
                        "type": "object"
                    },
                    "output_schema": {},
                    "transformations": {
                        "steps": []
                    },
                    "studio_id": tool_id
                }
            ],
            "partial_update": True
        }
        response = self._post(path, body=body)
        tool_response = self.retrieve_tool(tool_id)
        return tool_response

    def clone_tool(
        self,
        tool_id,
    ) -> Optional[Tool]: 
        path = "/studios/clone"
        body = {
            "studio_id": tool_id,
            "project": self._client.project,
            "region": self._client.region
        }
        response = self._post(path, body=body)
        cloned_tool_id = response.json().get("studio_id", None)
        if cloned_tool_id: 
            tool_response = self.retrieve_tool(cloned_tool_id)
            return tool_response
        else:
            return False

    def delete_tool(self, tool_id: str) -> bool:
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        response = self._post(path, body=body)
        return response.status_code == 200
    

class AsyncToolsManager(AsyncAPIResource):

    _client: AsyncRelevanceAI

    async def list_tools(
        self,
        max_results: Optional[int] = 100,
    ) -> List[AsyncTool]:
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
        response = await self._client.get(path, params=params)
        tools = [AsyncTool(client=self._client, **item) for item in response.json().get("results", [])]
        return tools
    
    async def retrieve_tool(self, tool_id: str) -> AsyncTool:
        path = f"studios/{tool_id}/get"
        response = await self._get(path)
        return AsyncTool(client=self._client, **response.json()["studio"])
    
    async def create_tool(
        self, 
        title: str, 
        description: str, 
        public: bool = False,
        params_schema = None,
        output_schema = None,
        transformations = None
    ) -> AsyncTool:
        tool_id = str(uuid.uuid4())  # Convert UUID to string
        path = "studios/bulk_update"
        body = {
            "updates": [
                {
                    "title": title,
                    "public": public,
                    "project": self._client.project,
                    "description": description,
                    "version": "latest",
                    "params_schema": {
                        "properties": {},
                        "required": [],
                        "type": "object"
                    },
                    "output_schema": {},
                    "transformations": {
                        "steps": []
                    },
                    "studio_id": tool_id
                }
            ],
            "partial_update": True
        }
        response = await self._post(path, body=body)
        tool_response = await self.retrieve_tool(tool_id)
        return tool_response
    
    async def clone_tool(
        self,
        tool_id: str,
    ) -> Optional[AsyncTool]:
        path = "/studios/clone"
        body = {
            "studio_id": tool_id,
            "project": self._client.project,
            "region": self._client.region
        }
        response = await self._post(path, body=body)
        cloned_tool_id = response.json().get("studio_id", None)
        if cloned_tool_id:
            tool_response = await self.retrieve_tool(cloned_tool_id)
            return tool_response
        else:
            return False

    async def delete_tool(self, tool_id: str) -> bool:
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        response = await self._post(path, body=body)
        return response.status_code == 200