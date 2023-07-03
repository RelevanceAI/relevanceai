from __future__ import annotations

import requests

from abc import abstractproperty

from typing import List
from pydantic import BaseModel, Field

from relevanceai.helpers import _handle_response
from relevanceai.constants import _BASE_URL
from relevanceai.auth import config
from relevanceai.types import JSONDict, JSONObject


class Step(BaseModel):
    transformation: str = Field(...)
    name: str = Field(None)
    params: JSONObject = Field(default_factory=lambda: {})
    output: JSONObject = Field(None)
    foreach: str = Field("")

    @property
    def param_keys(self):
        return set(self.dict()) - set(Step.schema()["properties"])

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = {}
        for k in self.param_keys:
            v = getattr(self, k, None)
            if v is not None:
                self.params[k] = v

    def to_dict(self, name: str = ""):
        json_obj = {
            "transformation": self.transformation,
            "name": self.name or name,
            "foreach": self.foreach,
            "params": self.params,
            "output": self.get_output_spec(),
        }
        return json_obj

    @abstractproperty
    def output_spec(self) -> List[str]:
        """
        Returns a list of strings that represent the possible outputs a step
        """
        raise NotImplementedError

    def get_output_spec(self):
        return {output: f"{{{{ {output} }}}}" for output in self.output_spec}

    def step_dict(self, name: str = "") -> JSONDict:
        json_obj = self.to_dict()
        json_obj["name"] = name or self.transformation
        if self.output:
            json_obj["output"] = self.output
        return json_obj

    def chain_dict(self, name: str, step_json: JSONDict) -> JSONDict:
        json_obj = {
            "studio_id": name,
            "public": False,
            "transformations": {"steps": [step_json]},
            "params_schema": {
                "properties": {},
            },
        }
        return json_obj

    def _trigger_json(self, return_state: bool = True, name: str = None) -> JSONDict:
        if name is None:
            name = f"Single Step - {self.transformation}"
        chain_dict = self.chain_dict(name, self.step_dict())
        return {
            "studio_id": name,
            "return_state": return_state,
            "studio_override": chain_dict,
            "state_override": {
                "steps": {},
                "params": {},
            },
            "params": {},
        }

    def run(self, return_state: bool = True, name: str = None):
        base_url = (
            f"{_BASE_URL.format(region=config.auth.region)}{config.auth.project}/"
        )
        response = requests.post(
            url=base_url + "trigger",
            json=self._trigger_json(return_state, name),
            headers=config.auth.headers,
        )
        return _handle_response(response)
