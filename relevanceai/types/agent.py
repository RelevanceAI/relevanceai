from pydantic import BaseModel
from typing import Optional, List, Dict, Any
class Agent(BaseModel):
    _id: str
    agent_id: str
    emoji: str
    machine_user_id: str
    project: str
    template: Optional[Dict[str, Any]] = None
    update_date_: str
    version: str
    action_behaviour: str
    actions: List[Dict[str, Any]]
    embeddable: Optional[bool] = None
    escalations: Optional[Dict[str, Any]] = None
    expiry_date_: Optional[str] = None
    is_scheduled_triggers_enabled: Optional[bool] = None
    knowledge: List[Any]
    max_job_duration: Optional[str] = None
    model: str
    name: str
    params: Optional[Dict[str, Any]] = None
    params_schema: Optional[Dict[str, Any]] = None
    public: Optional[bool] = None
    runner: Optional[Dict[str, Any]] = None
    starting_messages: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    title_prompt: Optional[str] = None

    class Config:
        extra = 'ignore'
