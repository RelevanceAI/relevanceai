from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

### Task Trigger 

class JobInfo(BaseModel):
    job_id: str
    studio_id: str

class TriggerTask(BaseModel):
    job_info: JobInfo
    conversation_id: str
    agent_id: str
    state: str
    def __repr__(self):
        return f"<Task - {self.conversation_id}>"
    
class ScheduledActionTrigger(BaseModel):
    trigger_id: str

### Task item
class Feedback(BaseModel):
    feedback: Optional[str] = Field(None, enum=["", "helpful", "unhelpful"])
    message_id: str

class Conversation(BaseModel):
    tags: Optional[Dict[str, Union[bool, Dict[str, str]]]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_region: Optional[Union[str, None]] = None
    agent_project: Optional[Union[str, None]] = None
    bookmark: Optional[bool] = False
    ignore_state_at: Optional[str] = None
    title: Optional[str] = None
    message_feedback: Optional[List[Feedback]] = None
    debug: Optional[Dict] = None
    has_errored: Optional[bool] = False
    params_from_previous_trigger: Optional[Dict] = None
    caller_agent: Optional[Dict] = None
    email_open_count: Optional[int] = 0
    is_debug_mode_task: Optional[bool] = False
    debug_mode_config_id: Optional[str] = None
    custom_params: Optional[Dict] = None
    enable_custom_params: Optional[bool] = False
    state_metadata: Optional[Dict] = None
    state: Optional[str] = Field(None, enum=["idle", "starting-up", "running", "pending-approval", "waiting-for-capacity", "cancelled", "timed-out", "escalated", "unrecoverable", "paused", "completed"])

class Metadata(BaseModel):
    _id: str
    project: str
    knowledge_set: str
    insert_date: str
    update_date: str
    expiry_date_: Optional[str] = None
    insert_datetime: Optional[str] = None
    update_datetime: Optional[str] = None
    model: Optional[str] = None
    external_id: Optional[str] = None
    source_dataset_id: Optional[str] = None
    conversation: Optional[Conversation] = None
    hidden: Optional[bool] = False
    type: Optional[str] = None

class TaskItem(BaseModel):
    knowledge_set: str
    knowledge_count: Optional[int] = None  
    knowledge_chunked_count: Optional[int] = None  
    knowledge_vectorized_count: Optional[int] = None  
    metadata: Metadata

    def __repr__(self):
        return f"<TaskItem \"{self.metadata.conversation.title}\" - {self.knowledge_set}>"
    
    def get_id(self):
        return self.knowledge_set

### Task Conversation Summary

class ErrorDetail(BaseModel):
    body: Optional[str] = None
    step_name: Optional[str] = None
    raw: Optional[str] = None

class CreditDetail(BaseModel):
    credits: float
    name: str
    num_units: Optional[float] = None
    multiplier: Optional[Union[float, None]] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_run_id: Optional[str] = None


class ExecutorDetail(BaseModel):
    type: str
    api_key_id: Optional[str] = None
    workflow_id: Optional[str] = None
    document_id: Optional[str] = None
    sync_id: Optional[str] = None
    job_id: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    email_message_id: Optional[str] = None


class TaskConversation(BaseModel):
    _id: Optional[str] = None
    version: Optional[str] = None
    studio_id: Optional[str] = None
    title: Optional[str] = None
    insert_date_: Optional[str] = None
    expiry_date_: Optional[str] = None
    status: Optional[str] = Field(
        None,
        description="Status of the workflow. Used for knowing when to send an email notification.",
        enum=["complete", "inprogress", "failed", "cancelled"],
    )
    errors: Optional[List[ErrorDetail]] = None
    execution_time: Optional[float] = None
    project: Optional[str] = None
    executor: Optional[ExecutorDetail] = None
    credits_used: Optional[List[CreditDetail]] = None
    max_job_duration: Optional[str] = Field(
        None,
        description="Switching this to hours tells our runner engine to run the job in a way suited for long runs.",
        enum=["hours", "minutes", "synchronous_seconds", "background_seconds"],
    )
    output_preview: Optional[dict] = None
    input_params: Optional[dict] = None
    cost: Optional[float] = None
    internal_job_id: Optional[str] = None
    internal_log_info: Optional[dict] = None

    def __repr__(self) -> str:
        return f"<TaskConversation \"{self.title}\" - {self.executor.conversation_id}>"