from .._client import RelevanceAI
from ..types.params import *
from ..types.tool import Tool, ToolOutput
from .._resource import SyncAPIResource
import json 
from typing import List

class Tools(SyncAPIResource):

    _client: RelevanceAI

    def retrieve(
        self, 
        tool_id: str
    ) -> Tool:
        path = f"studios/{tool_id}/get"
        body = None 
        params = None
        response = self._get(path=path, body=body, params=params)
        return Tool(**response.json()["studio"])
    
    def delete(
        self,
        tool_id: str
    ) -> bool:
        
        path = "studios/bulk_delete"
        body = {"ids": [tool_id]}
        params = None
        response = self._post(path=path, body=body, params=params)
        if response.status_code != 200:
            return True
        else:
            return False
        
    def list_tools(
        self,
    ) -> List[Tool]: 
        path = "studios/list"
        response = self._get(path=path)
        return [Tool(**item) for item in response.json().get("results", [])]

    def _list_params_as_string(
        self, 
        tool_id: str
    ) -> str:
        path = f"studios/{tool_id}/get"
        body = None 
        params = None
        response = self._get(path=path, body=body, params=params)
        return json.dumps(response.json()["studio"]["params_schema"]["properties"], indent=4)
        
    def _list_steps_as_string(
        self,
        tool_id: str
    ) -> str:
        path = f"studios/{tool_id}/get"
        body = None 
        params = None
        response = self._get(path=path, body=body, params=params)
        return json.dumps(response.json()["studio"]["transformations"]["steps"], indent=4)
    
    def trigger(
        self, 
        tool_id: str, 
        params: dict = None, 
    ) -> ToolOutput: 
        path = f"studios/{tool_id}/trigger" 
        body = {"params": params}
        response = self._post(path=path, body=body)
        return ToolOutput(**response.json())
        
    def bulk_trigger(
        self,
    ):
        pass