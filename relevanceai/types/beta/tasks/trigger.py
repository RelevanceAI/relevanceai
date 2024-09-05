
from typing import List, Union, Optional
from typing_extensions import Literal
from pydantic import BaseModel

class Trigger(BaseModel):
    trigger_id: str

