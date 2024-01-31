import json
import requests
from relevanceai._request import handle_response
from relevanceai import config
from relevanceai.auth import Auth
from relevanceai.params import Parameters


def create(name, description="", parameters={}, id=None, auth=None):
    """creates a new chain"""
    tool = Tool(
        name=name, description=description, parameters=parameters, id=id, auth=auth
    )
    return tool


def load(id, auth=None):
    """loads a chain via id"""
    if auth is None:
        auth = config.auth
    response = requests.get(
        f"{auth.url}/latest/studios/{auth.project}/{id}",
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
    tool = Tool(name="", description="", parameters={}, id=id, auth=auth)
    return tool


def load_from_json(filepath_or_json):
    if isinstance(filepath_or_json, str):
        with open(filepath_or_json, "r") as f:
            tool_json = json.load(f)
    else:
        tool_json = filepath_or_json
    tool = Tool(
        name=tool_json["title"],
        description=tool_json["description"],
        parameters=tool_json["params_schema"]["properties"],
        id=tool_json["studio_id"],
    )
    tool.add(tool_json["transformations"]["steps"])
    return tool


class Tool:
    def __init__(
        self,
        name: str,
        description: str = "",
        parameters={},
        id: str = None,
        auth: Auth = None,
    ):
        """
        Class for a Tool
        :param name: name of the tool
        :param description: description of the tool
        :param parameters: parameters of the tool
        :param id: id of the tool
        :param auth: auth object
        """
        self.name = name
        self.description = description
        self._parameters = parameters
        self.steps = []
        # generate random id if none provided
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
        tool_steps = [step.steps[0] for step in steps]
        unique_ids = []
        for step in tool_steps:
            if step["name"] in unique_ids:
                raise ValueError(
                    f"Duplicate step name {step['name']}, please rename the step name with Step(step_name=step_name)."
                )
            unique_ids.append(step["name"])
        return tool_steps

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
        data = {
            "title": self.name,
            "description": self.description,
            "version": "latest",
            "project": self.auth.project,
            "public": False,
            "state_mapping" : {
                "text" : "params.text"
            },
            "params_schema": {"properties": self.parameters.to_json()},
            "transformations": {"steps": self._transform_steps(self.steps)},
        }
        data["studio_id"] = self.id
        return data

    def deploy(self):
        url = f"{self.auth.url}/latest/studios"
        response = requests.post(
            f"{url}/bulk_update",
            json={"updates": [self._json()]},
            headers=self.auth.headers,
        )
        res = handle_response(response)
        print("Tool deployed successfully to id ", self.id)
        if self.random_id:
            print(
                "Your tool id is randomly generated, to ensure you are updating the same tool you should specify the id on rai.create(id=id) ",
            )
        print("\n=============Low Code Notebook================")
        print(
            f"You can share/visualize your tool as an app in our low code notebook here: https://app.relevanceai.com/notebook/{self.auth.region}/{self.auth.project}/{self.id}/app"
        )
        print("\n=============with Requests================")
        print("Here is an example of how to run the tool with API: ")
        print(
            f"""
import requests
requests.post({self.auth.url}/latest/studios/{self.id}/trigger_limited", json={{
    "project": "{self.auth.project}",
    "params": {{
        YOUR PARAMS HERE
    }}
}})
            """
        )
        print("\n=============with Python SDK================")
        print("Here is an example of how to run the tool with Python: ")
        print(
            f"""
import relevanceai as rai
tool = rai.load("{self.id}")
tool.run({{YOUR PARAMS HERE}})
            """
        )
        return self.id

    def to_json(self, filepath, return_json=False):
        if return_json:
            with open(filepath, "w") as f:
                json.dump(self._json(), f)
                print("Tool saved to ", filepath)
        else:
            return self._json()

    def reset(self):
        self.steps = []
