from __future__ import annotations

import uuid
import requests

from typing import List
from collections import UserList
from pydantic import BaseModel

from relevanceai.helpers import _handle_response
from relevanceai.constants import (
    _BASE_URL,
    _LOW_CODE_NOTEBOOK_BLURB,
    _RANDOM_ID_WARNING,
    _SCHEMA_KEYS,
)
from relevanceai.steps.base import Step
from relevanceai.auth import Auth, config
from relevanceai.types import JSONDict


class Chain(UserList):
    data: List[Step]

    def __init__(
        self,
        input: BaseModel,
        output: BaseModel,
        title: str = None,
        description: str = None,
        studio_id: str = None,
        public: bool = False,
        steps: List[Step] = None,
        auth: Auth = None,
    ):
        self.params = input
        self.output = output

        self.title = title
        self.description = description

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
        self.base_url = _BASE_URL.format(region=config.auth.region)

    def to_json(self, schema_props: JSONDict = None):
        if not schema_props:
            schema_props = self.params.schema()["properties"]
        json_obj = {
            "studio_id": self.studio_id,
            "public": self.public,
            "transformations": {
                "steps": [
                    step.step_dict(f"step{idx+1}") for idx, step in enumerate(self.data)
                ],
                "output": {
                    k: "{{ " + f"steps.{v}" + " }}"
                    for k, v in self.output.dict().items()
                },
            },
            "params_schema": {
                "properties": {
                    k: {sk: sv for sk, sv in v.items() if sk in _SCHEMA_KEYS}
                    for k, v in schema_props.items()
                },
            },
        }
        for key in ["title", "description"]:
            value = getattr(self, key, None)
            if value:
                json_obj[key] = value
        return json_obj

    def _trigger_json(self, return_state: bool = True):
        schema_props = self.params.schema()["properties"]
        return {
            "return_state": return_state,
            "studio_override": self.to_json(schema_props),
            "params": {k: v["default"] for k, v in schema_props.items()},
        }

    def run(self, return_state: bool = True):
        response = requests.post(
            url=self.base_url + f"{self.studio_id}/trigger",
            json=self._trigger_json(return_state),
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
