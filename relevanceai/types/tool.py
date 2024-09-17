from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ParamProperty(BaseModel):
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
    value: Optional[Any] = None
    order: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    max: Optional[int] = None
    min: Optional[int] = None
    items: Optional[Dict[str, Any]] = None

class ParamSchema(BaseModel):
    properties: Dict[str, ParamProperty]
    required: List[str]
    type: str

class TransformationStep(BaseModel):
    transformation: str
    name: str
    params: Dict[str, Any]

class Transformations(BaseModel):
    steps: List[TransformationStep]

class Tool(BaseModel):
    _id: str
    creator_first_name: str
    creator_last_name: str
    creator_user_id: str
    description: str
    output_schema: Dict[str, Any]
    params_schema: ParamSchema
    project: str
    public: bool
    state_mapping: Dict[str, str] = None
    studio_id: str
    title: str
    transformations: Transformations
    update_date_: str
    version: str
    machine_user_id: Optional[str] = None 

    class Config:
        extra = 'ignore'

    def __repr__(self):
        return f"<Tool \"{self.title}\" - {self.studio_id}>"

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class CreditUsage(BaseModel):
    credits: float
    name: str
    multiplier: Optional[int] = None
    num_units: Optional[int] = None
    tool_run_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None


class TransformedOutput(BaseModel):
    options: Optional[str] = None
    long_text: Optional[str] = None
    number: Optional[int] = None
    list: Optional[List[str]] = None
    json_list: Optional[List[Dict[str, Any]]] = None


class Output(BaseModel):
    transformed: TransformedOutput
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration: float
    credits_cost: float

class ToolOutput(BaseModel):
    status: str
    errors: List[str]
    output: Output
    credits_used: List[CreditUsage]
    executionTime: int
    cost: float
