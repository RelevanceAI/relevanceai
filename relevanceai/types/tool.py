from typing import List, Optional, Union, Dict
from pydantic import BaseModel

class DefaultOutputValues(BaseModel):
    original_key: str
    updated_key: Optional[str]
    value: Union[str, int, float, dict, list]

class TransformationStep(BaseModel):
    name: str
    transformation: str
    params: dict
    saved_params: Optional[dict] = None
    output: Optional[dict] = None
    default_output_values: Optional[List[DefaultOutputValues]] = None
    continue_on_error: Optional[bool] = None
    use_fallback_on_skip: Optional[bool] = None
    foreach: Optional[Union[str, List[str]]] = None
    if_condition: Optional[Union[str, bool, None]] = None
    display_name: Optional[str] = None

class Transformations(BaseModel):
    steps: List[TransformationStep]
    output: Optional[Union[dict, None]] = None

class WorkflowProperties(BaseModel):
    params: dict
    workflow_id: str
    host_type: Optional[str] = None
    dataset_id: Optional[str] = None
    version: Optional[str] = None

class TemplateTransformation(BaseModel):
    properties: WorkflowProperties
    depends_on: Optional[List[str]] = None
    repeat: Optional[str] = None
    repeat_index: Optional[float] = None
    if_condition: Optional[str] = None
    output_key: Optional[str] = None
    passthrough_email: Optional[bool] = None

class Template(BaseModel):
    transformations: Dict[str, TemplateTransformation]

class MetadataFieldOrder(BaseModel):
    field_order: Optional[List[str]] = None

class ParamsSchemaMetadata(BaseModel):
    content_type: Optional[str] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[str] = None
    api_selector_placeholder: Optional[str] = None
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = None
    relevance_only: Optional[bool] = None
    advanced: Optional[bool] = None
    placeholder: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = None
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[dict] = None
    enum: Optional[Union[List[dict], List[str]]] = None  
    bulk_run_input_source: Optional[str] = None
    agent_input_source: Optional[str] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[dict] = None
    language: Optional[str] = None
    key_value_input_opts: Optional[dict] = None
    knowledge_set_field_name: Optional[str] = None
    filters: Optional[List[dict]] = None
    oauth_permissions: Optional[List[dict]] = None
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = None
    oauth_account_provider: Optional[str] = None
    oauth_account_permission_type: Optional[str] = None
    scratchpad: Optional[dict] = None
    order: Optional[int] = None
    items: Optional[dict] = None

class ParamsSchema(BaseModel):
    metadata: Optional[MetadataFieldOrder] = None
    properties: Optional[Dict[str, ParamsSchemaMetadata]] = None

class Tags(BaseModel):
    type: Optional[str] = None
    categories: Optional[Dict[str, bool]] = None
    integration_source: Optional[str] = None

class State(BaseModel):
    params: Optional[dict] = None
    steps: Optional[Dict[str, dict]] = None

class Schedule(BaseModel):
    frequency: Optional[str] = None

class Metrics(BaseModel):
    views: Optional[int] = None
    executions: Optional[int] = None

class Tool(BaseModel):
    version: str
    project: str
    _id: str
    studio_id: str
    public: Optional[bool] = False
    insert_date_: Optional[str] = None 
    transformations: Transformations
    template: Optional[Template] = None
    update_date_: Optional[str] = None
    is_hidden: Optional[bool] = False
    tags: Optional[Tags] = None
    publicly_triggerable: Optional[bool] = False
    machine_user_id: Optional[str] = None
    creator_user_id: Optional[str] = None
    creator_first_name: Optional[str] = None
    creator_last_name: Optional[str] = None
    creator_display_picture: Optional[str] = None
    cover_image: Optional[str] = None
    emoji: Optional[str] = None
    params_schema: Optional[ParamsSchema] = None
    output_schema: Optional[dict] = None
    predicted_output: Optional[List[dict]] = None
    schedule: Optional[Schedule] = None
    state: Optional[State] = None
    title: Optional[str] = None
    description: Optional[str] = None
    prompt_description: Optional[str] = None
    state_mapping: Optional[dict] = None
    max_job_duration: Optional[str] = None
    metadata: Optional[dict] = None
    metrics: Optional[Metrics] = None
    share_id: Optional[str] = None

    def __repr__(self):
        return f"<Tool \"{self.title}\" - {self.studio_id}>"

###

class Error(BaseModel):
    body: Optional[str]
    step_name: Optional[str]
    raw: Optional[str]

class CreditsUsed(BaseModel):
    credits: float
    name: str
    num_units: Optional[float] = None
    multiplier: Optional[float] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_run_id: Optional[str] = None

class ToolOutput(BaseModel):
    output: dict
    state: Optional[dict] = None  
    status: str
    errors: List[Error]
    cost: Optional[float]
    credits_used: Optional[List[CreditsUsed]]
    executionTime: float

