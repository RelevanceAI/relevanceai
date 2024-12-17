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
    
    def update_settings(self):
        # title description
        pass

    # inputs / params 
    def update_params(self): 
        # {"updates":[{"version":"latest","_id":"587fc94b-50fa-4653-947f-a3c5d1a5e787_-_a56bc4da-591f-4925-89de-904a74d8003e_-_latest","creator_user_id":"3964a48d-ba44-4ff0-b9cd-0f17dd3c6f1a","description":"Description","output_schema":{},"params_schema":{"properties":{"text":{"type":"string","order":1,"title":"hello","metadata":{}}},"required":["text"],"type":"object"},"project":"587fc94b-50fa-4653-947f-a3c5d1a5e787","public":false,"studio_id":"a56bc4da-591f-4925-89de-904a74d8003e","title":"Blank","transformations":{"steps":[]},"update_date_":"2024-12-17T05:25:47.320Z","creator_first_name":"Ethan","creator_last_name":"Trang","params":{},"state_mapping":{"text":"params.text"}}],"partial_update":true}
        body_2 = {
            "updates": [
                {
                    "version": "latest",
                    "_id": "587fc94b-50fa-4653-947f-a3c5d1a5e787_-_a56bc4da-591f-4925-89de-904a74d8003e_-_latest",
                    "creator_user_id": "3964a48d-ba44-4ff0-b9cd-0f17dd3c6f1a",
                    "description": "Description",
                    "output_schema": {},
                    "params_schema": {
                        "properties": {
                            "text": {
                                "type": "string",
                                "order": 1,
                                "title": "hello",
                                "metadata": {}
                            },
                            "long_text": {
                                "type": "string",
                                "metadata": {
                                    "content_type": "long_text"
                                },
                                "order": 2,
                                "title": "long text",
                                "description": "more description"
                            }
                        },
                        "required": ["text", "long_text"],
                        "type": "object"
                    },
                    "project": "587fc94b-50fa-4653-947f-a3c5d1a5e787",
                    "public": False,
                    "studio_id": "a56bc4da-591f-4925-89de-904a74d8003e",
                    "title": "Blank",
                    "transformations": {
                        "steps": []
                    },
                    "update_date_": "2024-12-17T05:25:47.320Z",
                    "creator_first_name": "Ethan",
                    "creator_last_name": "Trang",
                    "params": {},
                    "state_mapping": {
                        "text": "params.text",
                        "long_text": "params.long_text"
                    }
                }
            ],
            "partial_update": True
        }
        body = {
            "updates": [
                {

                    "params": {},
                    "params_schema": {
                        "properties": {
                            "text": {
                                "type": "string",
                                "order": 1,
                                "title": "hello",
                                "metadata": {}
                            }
                        },
                        "required": ["text"],
                        "type": "object"
                    },
                    "state_mapping": {
                        "text": "params.text"
                    },
                    
                    "transformations": {
                        "steps": []
                    },
                    "output_schema": {},
                }
            ],
            "partial_update": True
        }

    # steps 
    def update_transformations(self):
        pass

    # output 
    def update_output_schema(self): 
        # last output / manual / write to agent metadata 
        pass

    def __repr__(self):
        return f'Tool(tool_id="{self.tool_id}", title="{self.metadata.title}")'