from __future__ import annotations

from .._client import RelevanceAI
from .._resource import SyncAPIResource
from ..types.tool import *
from typing import List, Optional, Dict
from ..types.params import *
from ..types.transformations import * 
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

    def get_transformations_schema(self) -> str:
        response = self._get(f"studios/{self.tool_id}/get")
        steps_schema = response.json()["studio"]["transformations"]["steps"]
        return json.dumps(steps_schema, indent=4)
    
    def update_metadata(
        self,
        title: str = None,
        description: str = None, 
        public: bool = None, 
    ):
        response = self._get(f"studios/{self.tool_id}/get")
        current_metadata = {
            "title": response.json()["studio"].get("title"),
            "description": response.json()["studio"].get("description"),
            "public": response.json()["studio"].get("public"),
        }

        updates = {
            "studio_id": self.tool_id,
            "title": title if title is not None else current_metadata["title"],
            "description": description if description is not None else current_metadata["description"],
            "public": public if public is not None else current_metadata["public"],
        }

        path = "studios/bulk_update"
        body = {
            "updates": [updates],
            "partial_update": True
        }
        response = self._post(path, body=body)
        return response.json()

    def update_params(
        self,
        params: Dict[str, ParamsBase],
    ) -> Tool:
        
        params_schema = {
            "properties": {},
            "required": [],
            "type": "object"
        }
        
        param_values = {
            field_name: param.value
            for field_name, param in params.items()
            if param.value is not None
        }

        for field_name, param in params.items():
            param_dict = param.model_dump(exclude_none=True)
            params_schema["properties"][field_name] = param_dict
            if param.required:
                params_schema["required"].append(field_name)

        state_mapping = {
            field_name: f"params.{field_name}" 
            for field_name in params.keys()
        }

        path = "studios/bulk_update"
        body = {
            "updates": [{
                "studio_id": self.tool_id,
                "params": param_values,
                "params_schema": params_schema,
                "state_mapping": state_mapping
            }],
            "partial_update": True
        }
        
        response = self._post(path, body=body)
        return response.json()
        
    def update_transformations(
        self,
        transformations: List[TransformationBase],
    ) -> dict:
        
        response = self._get(f"studios/{self.tool_id}/get")
        current_state = response.json()["studio"].get("state_mapping", {})

        state_mapping = {
            **current_state, 
            **{
                step.name: f"steps.{step.name}.output"
                for step in transformations
            }
        }
        
        transformation_config = {
            "steps": [
                transform.model_dump(exclude_none=True)
                for transform in transformations
            ]
        }
        
        path = "studios/bulk_update"
        body = {
            "updates": [{
                "studio_id": self.tool_id,
                "transformations": transformation_config,
                "state_mapping": state_mapping
            }],
            "partial_update": True
        }
        
        response = self._post(path, body=body)
        return response.json()
    
    def update_outputs(
        self, 
        last_step: bool = True,
        output_mapping: Optional[dict] = None,
        output_schema_properties: Optional[dict] = None,
    ) -> dict: 
        response = self._get(f"studios/{self.tool_id}/get")
        current_steps = response.json()["studio"].get("transformations", {}).get("steps", [])
        
        transformations = {"steps": current_steps}
        transformations["output"] = output_mapping if not last_step else None

        updates = {
            "studio_id": self.tool_id,
            "transformations": transformations,
        }

        if output_schema_properties:
            updates["output_schema"] = {
                "properties": output_schema_properties
            }

        path = "studios/bulk_update"
        body = {
            "updates": [updates],
            "partial_update": True
        }

        response = self._post(path, body=body)
        return response.json()
    
    def get_link(self):
        return f"https://app.relevanceai.com/agents/{self._client.region}/{self._client.project}/{self.tool_id}"

    def __repr__(self):
        return f'Tool(tool_id="{self.tool_id}", title="{self.metadata.title}")'