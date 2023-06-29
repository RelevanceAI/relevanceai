from pydantic import BaseModel, Field
from relevanceai.types import JSONObject


class Step(BaseModel):
    transformation: str = Field(...)
    name: str = Field(None)
    params: JSONObject = Field(...)
    output: JSONObject = Field(None)
    foreach: str = Field("")
    strip_linebreaks: bool = Field(True)

    def to_json(self, name: str):
        json_obj = {
            "transformation": self.transformation,
            "name": self.name or name,
            "foreach": self.foreach,
            "params": {
                **self.params,
                "strip_linebreaks": self.strip_linebreaks,
            },
        }
        if self.output:
            json_obj["output"] = self.output
        return json_obj
