from __future__ import annotations

import uuid
import json
import requests

from collections import UserList
from pydantic import BaseModel
from typing import List

from relevanceai.auth import Auth, config
from relevanceai.types import JSONObject
from relevanceai._request import _handle_response
from relevanceai.steps._base import Step


def create(name, description="", parameters={}, id=None, auth=None):
    chain = Chain(
        name=name, description=description, parameters=parameters, id=id, auth=auth
    )
    return chain


def load(id, auth=None):
    if auth is None:
        auth = config.auth
    response = requests.get(
        f"https://api-{auth.region}.stack.tryrelevance.com/latest/studios/{auth.project}/{id}",
        json={
            "filters": [
                {
                    "field": "studio_id",
                    "condition": "==",
                    "condition_value": id,
                    "filter_type": "exact_match",
                },
                {
                    "field": "project",
                    "condition": "==",
                    "condition_value": auth.project,
                    "filter_type": "exact_match",
                },
            ]
        },
    )
    res = _handle_response(response)
    chain = Chain(name="", description="", parameters={}, id=id, auth=auth)
    return chain


_SCHEMA_KEYS = {"title", "type", "description"}
_BASE_URL = "https://api-{region}.stack.tryrelevance.com/latest/studios/"
_RANDOM_ID_WARNING = "Your studio id is randomly generated, to ensure you are updating the same chain you should specify the id on rai.create(id=id) "
_LOW_CODE_NOTEBOOK_BLURB = """
=============Low Code Notebook================
You can share/visualize your chain as an app in our low code notebook here: https://chain.relevanceai.com/notebook/{region}/{project}/{id}/app

===============With Requests==================
Here is an example of how to run the chain with API: 
import requests
requests.post(https://api-{region}.stack.tryrelevance.com/latest/studios/{id}/trigger_limited", json={{
    "project": "{project}",
    "params": {{
        YOUR PARAMS HERE
    }}
}})

===============With Python SDK================      
Here is an example of how to run the chain with Python: 
import relevanceai as rai
chain = rai.load("{id}")
chain.run({{YOUR PARAMS HERE}})
            
"""


class Chain(UserList):
    data: List[Step]

    def __init__(
        self,
        input: BaseModel,
        output: BaseModel,
        studio_id: str = None,
        public: bool = False,
        steps: List[Step] = None,
        auth: Auth = None,
    ):
        self.params = input
        self.output = output

        if steps:
            self.data = steps
        else:
            self.data = []

        if not studio_id:
            studio_id = str(uuid.uuid4())
            self.random_id = True
        else:
            self.random_id = False
        self.studio_id = studio_id

        self.public = public

        self.auth: Auth = config.auth if auth is None else auth
        self.base_url = _BASE_URL.format(region=self.auth.region)

    @property
    def params_schema(self) -> JSONObject:
        return json.loads(self.params.schema_json())

    def to_json(self):
        schema_props = self.params_schema["properties"]
        json_obj = {
            "studio_id": self.studio_id,
            "public": self.public,
            "transformations": {
                "steps": [
                    step.to_json(f"step{idx+1}") for idx, step in enumerate(self.data)
                ],
                "output": {
                    k: f"{{{ v }}}" for k, v in json.loads(self.output.json()).items()
                },
            },
            "params_schema": {
                "properties": {
                    k: {sk: sv for sk, sv in v.items() if sk in _SCHEMA_KEYS}
                    for k, v in schema_props.items()
                },
            },
        }
        return json_obj

    def run(self, return_state: bool = True):
        schema_props = self.params_schema["properties"]
        response = requests.post(
            url=self.base_url + "trigger",
            json={
                "return_state": return_state,
                "studio_override": self.to_json(),
                "params": {k: v["default"] for k, v in schema_props.items()},
            },
            headers=self.auth.headers,
        )
        return _handle_response(response)

    def reset(self):
        self.data = []

    def deploy(self):
        response = requests.post(
            url=self.base_url + "bulk_update",
            json={"updates": [self.to_json()]},
            headers=self.auth.headers,
        )
        if not response.ok:
            raise ValueError(_handle_response(response))
        print(f"Studio deployed successfully with id: `{self.studio_id}`")
        if self.random_id:
            print(_RANDOM_ID_WARNING)
        print(
            _LOW_CODE_NOTEBOOK_BLURB.format(
                region=self.auth.region,
                project=self.auth.project,
                id=self.studio_id,
            )
        )
        return self.studio_id
