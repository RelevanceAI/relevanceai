
from ..types.params import *
from ..types.tool import Tool, ToolOutput
from .._resource import SyncAPIResource
from functools import cached_property
import json 

class Tools(SyncAPIResource):

    def create(
        self,
    ) -> Tool: 
        pass

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
        pass

    def update_params(
        self,
        tool_id: str
    ):  
        # todo: add, delete, reorder params
        pass

    def update_steps(
        self,
        tool_id: str
    ):
        # todo: add, delete, reorder params
        pass

    def _list_params_as_string(
        self, 
        tool_id: str
    ) -> str:
        """Returns a parameters template"""
        path = f"studios/{tool_id}/get"
        body = None 
        params = None
        response = self._get(path=path, body=body, params=params)
        return json.dumps(response.json()["studio"]["params_schema"]["properties"], indent=4)
        
    def _list_steps_as_string(
        self,
        tool_id: str
    ) -> str:
        """Returns a transformations template"""
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