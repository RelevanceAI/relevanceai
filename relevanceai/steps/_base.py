import requests
from relevanceai import config
from relevanceai.auth import Auth
from relevanceai._request import handle_response
from relevanceai.params import Parameters, ParamBase


class StepBase:
    def __init__(
        self, name="step", description="a step", parameters={}, id="new", auth=None
    ):
        self.name = name
        self.description = description
        if isinstance(parameters, Parameters):
            self.parameters = parameters.to_json()
        elif isinstance(parameters, ParamBase):
            self.parameters = parameters.to_json()
        else:
            self.parameters = parameters
        self.id = id
        self.auth: Auth = config.auth if auth is None else auth

    def _trigger_json(
        self, values: dict = {}, return_state: bool = True, public: bool = False
    ):
        return {
            "studio_id": self.id,
            "return_state": return_state,
            "params": values,
            "state_override": {
                "steps": {},
                "params": values,
            },
            "studio_override": {
                "studio_id": self.id,
                "public": public,
                "transformations": {"steps": self.steps},
                "params_schema": {"properties": self.parameters},
            },
        }

    def run(self, parameters={}, full_response: bool = False):
        url = f"{self.auth.url}/latest/studios/{self.auth.project}"
        response = requests.post(
            f"{url}/trigger",
            json=self._trigger_json(parameters),
            headers=self.auth.headers,
        )
        res = handle_response(response)
        if isinstance(res, dict):
            if ("errors" in res and res["errors"]) or full_response:
                return res
            elif "output" in res:
                return res["output"]
        return res

    def _json(self):
        return {
            "title": self.name,
            "description": self.description,
            "version": "latest",
            "project": self.auth.project,
            "studio_id": self.id,
            "public": False,
            "params_schema": {"properties": self.parameters},
            "transformations": {"steps": self.steps},
        }

    def deploy(self):
        url = f"{self.auth.url}/latest/studios"
        response = requests.post(
            f"{url}/bulk_update",
            json={"updates": [self._json()]},
            headers=self.auth.headers,
        )
        return handle_response(response)
