from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

class Feedback(Enum):
    field_ = ''
    helpful = 'helpful'
    unhelpful = 'unhelpful'


class MessageFeedbackItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    feedback: Feedback
    message_id: str


class CallerAgent(BaseModel):
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None


class UnrecoverableErrorType(Enum):
    max_tool_retries = 'max-tool-retries'


class StateMetadata(BaseModel):
    unrecoverable_error_type: Optional[UnrecoverableErrorType] = None


class State(Enum):
    idle = 'idle'
    starting_up = 'starting-up'
    running = 'running'
    pending_approval = 'pending-approval'
    waiting_for_capacity = 'waiting-for-capacity'
    cancelled = 'cancelled'
    timed_out = 'timed-out'
    escalated = 'escalated'
    unrecoverable = 'unrecoverable'
    paused = 'paused'
    completed = 'completed'
    errored_pending_approval = 'errored-pending-approval'


class Conversation(BaseModel):
    tags: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_region: Optional[str] = None
    agent_project: Optional[str] = None
    bookmark: Optional[bool] = None
    ignore_state_at: Optional[str] = Field(
        None,
        description='Time from which we can ignore the "state" field, and trigger the agent anyway. Prevents conversations from being permanently locked if an agent job stalls.',
    )
    title: Optional[str] = None
    message_feedback: Optional[List[MessageFeedbackItem]] = None
    debug: Optional[Dict[str, Any]] = None
    has_errored: Optional[bool] = None
    params_from_previous_trigger: Optional[Dict[str, Any]] = None
    caller_agent: Optional[CallerAgent] = None
    email_open_count: Optional[float] = Field(
        0,
        description='Sum total number of times all emails in this conversation have been opened',
    )
    is_debug_mode_task: Optional[bool] = None
    debug_mode_config_id: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None
    enable_custom_params: Optional[bool] = None
    state_metadata: Optional[StateMetadata] = None
    state: Optional[State] = None


class Type(Enum):
    num_tokens = 'num_tokens'


class SplitChunk(Enum):
    token = 'token'
    sentence = 'sentence'


class SplitMethod(BaseModel):
    type: Type
    num_tokens: float = Field(..., description='Maximum number of tokens per chunk')
    split_chunk: Optional[SplitChunk] = 'sentence'


class FieldMetadata(BaseModel):
    alias: Optional[str] = None
    should_vectorize: Optional[bool] = None


class Chain(BaseModel):
    project: str
    region: str
    studio_id: str
    version: Optional[str] = None


class Type1(Enum):
    dataset_field = 'dataset_field'
    value = 'value'


class ParamMapping(BaseModel):
    type: Optional[Type1] = None
    value: Optional[Any] = None


class ToolColumns(BaseModel):
    chain: Optional[Chain] = Field(
        None, description='Metadata for the chain (tool) used in the column.'
    )
    paramMapping: Optional[Dict[str, ParamMapping]] = Field(
        None,
        description='A map of tool params to their values. Params can be inferred from table fields or by entering values manually.',
    )
    outputMapping: Optional[Dict[str, Union[str, bool]]] = Field(
        None,
        description='A map of tool output keys to column names to overwrite. Tools can overwrite existing columns.',
    )
    uid: Optional[str] = Field(None, description='The UUID for the column.')


class TableMetadata(BaseModel):
    tool_columns: Optional[Dict[str, ToolColumns]] = Field(
        None,
        description='A map of tool column UUIDs to their corresponding column metadata.',
    )


class LastJobInfo(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    job_id: str
    studio_id: str


class Status(Enum):
    unvectorized = 'unvectorized'
    vectorizing = 'vectorizing'
    vectorized = 'vectorized'
    failed = 'failed'


class VectorizingInfo(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    last_job_info: Optional[LastJobInfo] = None
    status: Optional[Status] = None


class Metadata(BaseModel):
    field_id: str = Field(..., alias='_id')
    project: str
    knowledge_set: str
    insert_date: Optional[str] = None
    update_date: Optional[str] = None
    expiry_date_: Optional[Any] = None
    insert_datetime: Optional[Any] = None
    update_datetime: Optional[Any] = None
    model: Optional[str] = None
    external_id: Optional[str] = None
    source_dataset_id: Optional[str] = None
    conversation: Optional[Conversation] = None
    hidden: Optional[bool] = None
    type: Optional[str] = None
    split_method: Optional[SplitMethod] = None
    field_metadata: Optional[Dict[str, FieldMetadata]] = None
    table_metadata: Optional[TableMetadata] = Field(
        None, description='Metadata for knowledge sets in the table view.'
    )
    vectorizing_info: Optional[VectorizingInfo] = None


class KnowledgeSet(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    knowledge_set: str
    knowledge_count: Optional[float] = None
    knowledge_chunked_count: Optional[float] = None
    knowledge_vectorized_count: Optional[float] = None
    metadata: Optional[Metadata] = None

    def __repr__(self): 
        return f"KnowledgeSet(knowledge_set=\"{self.knowledge_set}\")"

class KnowledgeRow(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    field_id: str = Field(..., alias='_id')
    project: str
    knowledge_set: str
    document_id: str
    alias: str
    data: Dict[str, Any]
    tags: Optional[Dict[str, Any]] = None
    insert_date_: Optional[Any] = None
    update_date_: Optional[Any] = None
    expiry_date_: Optional[Any] = None
    is_chunked: Optional[bool] = None
    is_vectorized: Optional[bool] = None

    def __repr__(self): 
        return f"KnowledgeRow(field_id=\"{self.field_id}\")"
