from typing import Any

class ParamBase:
    def __init__(self, name):
        self.name = name
        self.json = {name: {}}
    
    def to_json(self):
        return self.json

class Parameters:
    def __init__(self, parameters):
        self.parameters = parameters

    def _check_param(self, param):
        if isinstance(param, dict):
            return param
        elif isinstance(param, ParamBase):
            return param.json
        else:
            raise ValueError(
                "Parameters must be a ParamBase instances or a dictionary"
            )

    def to_json(self):
        if isinstance(self.parameters, list):
            dict_params = {}
            for p in self.parameters:
                dict_params.update(self._check_param(p))
            return dict_params
        elif isinstance(self.parameters, dict):
            return self.parameters
        else:
            return self.parameters.json

    def _format_name(self, name:str):
        return "{{" + name + "}}"


class StringParam(ParamBase):
    def __init__(self, name, long: bool = False, title="Text Input", description=""):
        super().__init__(name)
        self.title = title
        self.description = description
        self.long = long
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "string",
            }
        }
        if self.long:
            self.json[name]["metadata"] = {"content_type": "long_text"}


class NumberParam(ParamBase):
    def __init__(
        self,
        name,
        max: int = None,
        min: int = None,
        title="Number Input",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.max = max
        self.min = min
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "number",
            }
        }
        if self.max:
            self.json[name]["max"] = max
        if self.min:
            self.json[name]["min"] = min


class OptionsParam:
    def __init__(
        self,
        name,
        options,
        title="Options",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.options = options
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "string",
                "enum": options,
            }
        }


class StringListParam(ParamBase):
    def __init__(
        self,
        name,
        title="Text List Input",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "array",
                "items": {"type": "string"},
            }
        }


class JsonParam:
    def __init__(
        self,
        name,
        title="JSON Input",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "object",
            }
        }


class JsonListParam(ParamBase):
    def __init__(
        self,
        name,
        title="JSON List Input",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "array",
                "items": {"type": "object"},
            }
        }


class FileParam(ParamBase):
    def __init__(
        self,
        name,
        title="File Input",
        description="",
    ):
        self.name = name
        self.title = title
        self.description = description
        self.json = {
            name: {
                "title": title,
                "description": description,
                "type": "string",
                "metadata": {"content_type": "file", "accepted_file_types": []},
            }
        }
