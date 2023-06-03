import json
import requests
from relevanceai._request import handle_response
from relevanceai import config
from relevanceai.auth import Auth
from relevanceai.params import Parameters


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
    res = handle_response(response)
    chain = Chain(name="", description="", parameters={}, id=id, auth=auth)
    return chain


def load_from_json(filepath_or_json):
    if isinstance(filepath_or_json, str):
        with open(filepath_or_json, "r") as f:
            chain_json = json.load(f)
    else:
        chain_json = filepath_or_json
    chain = Chain(
        name=chain_json["title"],
        description=chain_json["description"],
        parameters=chain_json["params_schema"]["properties"],
        id=chain_json["studio_id"],
    )
    chain.add(chain_json["transformations"]["steps"])
    return chain


class Chain:
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters={},
        id: str = None,
        auth: Auth = None,
    ):
        self.name = name
        self.description = description
        self._parameters = parameters
        self.steps = []
        # generate random id if none
        self.random_id = False
        if id is None:
            import uuid

            id = str(uuid.uuid4())
            self.random_id = True
        self.id = id
        self.auth: Auth = config.auth if auth is None else auth

    @property
    def parameters(self):
        return Parameters(self._parameters)

    params = parameters

    def add(self, steps):
        if isinstance(steps, list):
            self.steps.extend(steps)
        else:
            self.steps.append(steps)

    def _transform_steps(self, steps):
        chain_steps = [step.steps[0] for step in steps]
        unique_ids = []
        for step in chain_steps:
            if step["name"] in unique_ids:
                raise ValueError(
                    f"Duplicate step name {step['name']}, please rename the step name with Step(step_name=step_name)."
                )
            unique_ids.append(step["name"])
        return chain_steps

    def _trigger_json(
        self, values: dict = {}, return_state: bool = True, public: bool = False
    ):
        data = {
            "return_state": return_state,
            "studio_override": {
                "public": public,
                "transformations": {"steps": self._transform_steps(self.steps)},
                "params_schema": {"properties": self.parameters.to_json()},
            },
            "params": values,
        }
        data["studio_id"] = self.id
        data["studio_override"]["studio_id"] = self.id
        return data

    def run(self, parameters={}, full_response: bool = False):
        url = f"https://api-{self.auth.region}.stack.tryrelevance.com/latest/studios/{self.auth.project}"
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
        data = {
            "title": self.name,
            "description": self.description,
            "version": "latest",
            "project": self.auth.project,
            "public": False,
            "params_schema": {"properties": self.parameters.to_json()},
            "transformations": {"steps": self._transform_steps(self.steps)},
        }
        data["studio_id"] = self.id
        return data

    def deploy(self):
        url = f"https://api-{self.auth.region}.stack.tryrelevance.com/latest/studios"
        response = requests.post(
            f"{url}/bulk_update",
            json={"updates": [self._json()]},
            headers=self.auth.headers,
        )
        res = handle_response(response)
        print("Studio deployed successfully to id ", self.id)
        if self.random_id:
            print(
                "Your studio id is randomly generated, to ensure you are updating the same chain you should specify the id on rai.create(id=id) ",
            )
        print("\n=============Low Code Notebook================")
        print(
            f"You can share/visualize your chain as an app in our low code notebook here: https://chain.relevanceai.com/notebook/{self.auth.region}/{self.auth.project}/{self.id}/app"
        )
        print("\n=============with Requests================")
        print("Here is an example of how to run the chain with API: ")
        print(
            f"""
import requests
requests.post(https://api-{self.auth.region}.stack.tryrelevance.com/latest/studios/{self.id}/trigger_limited", json={{
    "project": "{self.auth.project}",
    "params": {{
        YOUR PARAMS HERE
    }}
}})
            """
        )
        print("\n=============with Python SDK================")
        print("Here is an example of how to run the chain with Python: ")
        print(
            f"""
import relevanceai as rai
chain = rai.load("{self.id}")
chain.run({{YOUR PARAMS HERE}})
            """
        )
        return self.id

    def to_json(self, filepath, return_json=False):
        if return_json:
            with open(filepath, "w") as f:
                json.dump(self._json(), f)
                print("Chain saved to ", filepath)
        else:
            return self._json()

    def reset(self):
        self.steps = []
