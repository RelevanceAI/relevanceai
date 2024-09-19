from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class MessageFeedback(BaseModel):
    feedback: str
    message_id: str

class CallerAgent(BaseModel):
    agent_id: str
    conversation_id: str
    project: str
    region: str

class Conversation(BaseModel):
    tags: Optional[Dict]
    user_id: Optional[str]
    agent_id: Optional[str]
    agent_region: Optional[Union[str, None]]
    agent_project: Optional[Union[str, None]]
    bookmark: Optional[bool]
    ignore_state_at: Optional[str] = Field(
        None,
        description='Time from which we can ignore the "state" field, and trigger the agent anyway. Prevents conversations from being permanently locked if an agent job stalls.'
    )
    title: Optional[str]
    message_feedback: Optional[List[MessageFeedback]]
    debug: Optional[Dict]
    has_errored: Optional[bool]
    params_from_previous_trigger: Optional[Dict]
    caller_agent: Optional[CallerAgent]
    email_open_count: Optional[int] = Field(0, description='Sum total number of times all emails in this conversation have been opened')
    is_debug_mode_task: Optional[bool]
    debug_mode_config_id: Optional[str]
    custom_params: Optional[Dict]
    enable_custom_params: Optional[bool]
    state_metadata: Optional[Dict]
    state: Optional[str]

class Metadata(BaseModel):
    _id: str
    project: str
    knowledge_set: str
    insert_date: str
    update_date: str
    expiry_date_: Optional[dict] = None
    insert_datetime: Optional[str]  # Changed from dict to str
    update_datetime: Optional[str]  # Changed from dict to str
    model: str
    external_id: Optional[str] = None
    source_dataset_id: Optional[str] = None
    conversation: Optional[Conversation] = None

class SplitMethod(BaseModel):
    type: Optional[str]
    num_tokens: Optional[int] = Field(500, description='Maximum number of tokens per chunk')
    split_chunk: Optional[str] = Field('sentence')

class Chain(BaseModel):
    project: str
    region: str
    studio_id: str
    version: Optional[str]

class ParamMapping(BaseModel):
    type: Optional[str]
    value: Optional[Union[str, dict]]

class ToolColumns(BaseModel):
    chain: Chain
    paramMapping: Optional[ParamMapping]
    outputMapping: Optional[Dict]
    uid: str

class FieldMetadata(BaseModel):
    alias: Optional[str]
    should_vectorize: Optional[bool]

class TableMetadata(BaseModel):
    tool_columns: Optional[Dict]

class VectorizingInfo(BaseModel):
    last_job_info: Optional[dict]
    status: str

class KnowledgeSet(BaseModel):
    knowledge_set: str
    knowledge_count: float
    knowledge_chunked_count: float
    knowledge_vectorized_count: float
    metadata: Metadata
    hidden: Optional[bool] = None
    type: Optional[str] = None
    split_method: Optional[SplitMethod] = None
    field_metadata: Optional[Dict] = None
    table_metadata: Optional[TableMetadata] = None
    vectorizing_info: Optional[VectorizingInfo] = None

    def __repr__(self): 
        return f"<KnowledgeItem \"{self.knowledge_set}\">"

class KnowledgeRow(BaseModel):
    id: str = Field(alias="_id")  # Alias to accept _id as input
    project: str
    knowledge_set: str
    document_id: str
    alias: str
    data: Dict
    tags: Optional[Dict] = None
    insert_date_: Optional[str] = None
    update_date_: Optional[str] = None
    expiry_date_: Optional[dict] = None
    is_chunked: Optional[bool] = None
    is_vectorized: Optional[bool] = None

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True

