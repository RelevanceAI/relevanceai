from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.tool import *
from typing import List, Optional
import json


class Tool(SyncAPIResource):
    _client: RelevanceAI

    def __init__(self, client: RelevanceAI, **metadata):
        super().__init__(client=client)
        self.metadata = ToolType(**metadata)
        self.tool_id = self.metadata.studio_id

    def update(
        self,
        updates: dict,
        partial_update: Optional[bool] = True,
    ) -> dict:
        path = "studios/bulk_update"
        body = {
            "partial_update": partial_update,
            "updates": [updates | {"studio_id": self.tool_id}],
        }
        response = self._client.post(path, body=body)
        return response.json()

    def trigger(
        self,
        params: dict | None = None,
    ):
        path = f"studios/{self.tool_id}/trigger_limited"
        body = {"params": params, "project": self._client.project}
        response = self._post(path=path, body=body)
        return ToolOutput(**response.json())

    def get_params_schema(self) -> str:
        response = self._get(f"studios/{self.tool_id}/get")
        params_schema = response.json()["studio"]["params_schema"]["properties"]
        return json.dumps(params_schema, indent=4)

    def get_steps(self) -> str:
        response = self._get(f"studios/{self.tool_id}/get")
        transformations = response.json()["studio"]["transformations"]
        steps = transformations["steps"]
        return json.dumps(steps, indent=4)
    
    # steps 
    def add_python_step(self):
        pass

    def remove_python_step(self):
        pass
    
    def update_python_step(self): 
        pass

    # inputs / params 

    def add_param(self): 
        # agent setting, default value
        pass

    def remove_param(self): 
        pass

    # output 
    def configure_output(self): 
        # last output / manual / write to agent metadata 
        pass

    def __repr__(self):
        return f'Tool(tool_id="{self.tool_id}", title="{self.metadata.title}")'