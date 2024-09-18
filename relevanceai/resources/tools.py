from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.tool import Tool, ToolOutput
from typing import List
import json 

class Tools(SyncAPIResource):

    _client: RelevanceAI

    # todo: some tools are helper tools
    def list_tools(self) -> List[Tool]:
        response = self._client.get("studios/list")
        tools = [Tool(**item) for item in response.json().get("results", [])]
        for tool in tools:
            try:
                tool.studio_id = tool.metadata["source_studio_id"]
            except KeyError:
                pass
        return tools

    def retrieve_tool(self, tool_id: str) -> Tool:
        path = f"studios/{tool_id}/get"
        response = self._get(path)
        return Tool(**response.json()["studio"])

    def trigger_tool(
        self,
        tool_id: str,
        params: dict | None = None,
    ) -> ToolOutput:
        path = f"studios/{tool_id}/trigger"
        body = {"params": params or {}}
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
