from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, confloat, constr


class Feedback(Enum):
    field_ = ''
    helpful = 'helpful'
    unhelpful = 'unhelpful'

class MessageFeedbackItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
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
        protected_namespaces=() 
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
        protected_namespaces=() 
    )
    last_job_info: Optional[LastJobInfo] = None
    status: Optional[Status] = None


class TaskMetadata(BaseModel):
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


class Task(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    knowledge_set: str
    knowledge_count: Optional[float] = None
    knowledge_chunked_count: Optional[float] = None
    knowledge_vectorized_count: Optional[float] = None
    metadata: Optional[TaskMetadata] = None

    def __repr__(self): 
        return f"Task(knowledge_set=\"{self.knowledge_set}\")"
    
    def get_id(self): 
        return self.knowledge_set


class Region(Enum):
    field_1e3042 = '1e3042'
    f1db6c = 'f1db6c'
    d7b62b = 'd7b62b'
    bcbe5a = 'bcbe5a'
    field_ = ''


class Template(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    agent_id: str
    region: Optional[Region] = None
    project: Optional[str] = None


class Origin(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    root_agent_id: str
    region: Optional[str] = None
    project: Optional[str] = None


class MaxJobDuration(Enum):
    hours = 'hours'
    minutes = 'minutes'
    synchronous_seconds = 'synchronous_seconds'
    background_seconds = 'background_seconds'


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    knowledge_set: str


class Metadata1(BaseModel):
    field_order: Optional[List[str]] = None


class ContentType(Enum):
    json = 'json'
    json_list = 'json_list'
    long_text = 'long_text'
    short_text = 'short_text'
    file_url = 'file_url'
    file_urls = 'file_urls'
    llm_prompt = 'llm_prompt'
    speech = 'speech'
    code = 'code'
    dataset_id = 'dataset_id'
    knowledge_set = 'knowledge_set'
    markdown = 'markdown'
    chain_id = 'chain_id'
    chain_params = 'chain_params'
    file_to_text = 'file_to_text'
    file_to_text_llm_friendly = 'file_to_text_llm_friendly'
    memory_optimizer = 'memory_optimizer'
    memory = 'memory'
    table = 'table'
    agent_id = 'agent_id'
    api_key = 'api_key'
    key_value_input = 'key_value_input'
    knowledge_editor = 'knowledge_editor'
    oauth_account = 'oauth_account'
    datetime = 'datetime'
    api_selector = 'api_selector'
    llm_model_selector = 'llm_model_selector'
    colour_picker = 'colour_picker'
    conditional = 'conditional'
    external_field = 'external_field'
    tool_approval = 'tool_approval'


class ApiSelectorType(Enum):
    finetuning_model_select = 'finetuning_model_select'
    elevenlabs_voice_selector = 'elevenlabs_voice_selector'
    elevenlabs_model_selector = 'elevenlabs_model_selector'
    heygen_voice_selector = 'heygen_voice_selector'
    heygen_avatar_selector = 'heygen_avatar_selector'
    llm_model_selector = 'llm_model_selector'
    blandai_voice_selector = 'blandai_voice_selector'
    d_id_voice_selector = 'd_id_voice_selector'
    d_id_avatar_selector = 'd_id_avatar_selector'
    vapi_custom_phone_number_selector = 'vapi_custom_phone_number_selector'
    vapi_custom_assistant_selector = 'vapi_custom_assistant_selector'
    webflow_collections = 'webflow_collections'


class ValueSuggestionChain(BaseModel):
    url: str
    project_id: str
    output_key: Optional[str] = 'value'


class EnumItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    description: str
    value: str
    group_name: Optional[str] = None


class BulkRunInputSource(Enum):
    field_ = ''
    field_DOCUMENT = '$DOCUMENT'
    field_FIELD_PARAM_MAPPING = '$FIELD_PARAM_MAPPING'


class AgentInputSource(Enum):
    conversation_id = 'conversation_id'
    agent_id = 'agent_id'


class Language(Enum):
    python = 'python'
    javascript = 'javascript'
    html = 'html'


class Header(BaseModel):
    hide: Optional[bool] = Field(None, description='Whether to hide all headers.')
    key: Optional[str] = Field(
        None, description='The header displayed above the key column.'
    )
    value: Optional[str] = Field(
        None, description='The header displayed above the value column.'
    )


class Placeholder(BaseModel):
    hide: Optional[bool] = Field(None, description='Whether to hide all placeholders.')
    key: Optional[str] = Field(
        None, description='The placeholder to display in each cell of the key column.'
    )
    value: Optional[str] = Field(
        None, description='The placeholder to display in each cell of the value column.'
    )


class KeyValueInputOpts(BaseModel):
    header: Optional[Header] = Field(
        None, description='Set headers to display above the key and/or value columns.'
    )
    placeholder: Optional[Placeholder] = Field(
        None,
        description='Set placeholder values to display in the key and/or value columns.',
    )
    addButtonText: Optional[str] = Field(
        None,
        description="The text displayed in the 'Add' button that inserts a new pair.",
    )


class Strict(Enum):
    must = 'must'
    should = 'should'
    must_or = 'must_or'


class FilterType(Enum):
    text_match = 'text_match'
    word_match = 'word_match'
    term = 'term'
    terms = 'terms'
    text = 'text'
    texts = 'texts'
    match = 'match'
    contains = 'contains'
    substring = 'substring'
    class_ = 'class'
    category = 'category'
    exact_match = 'exact_match'
    classes = 'classes'
    categories = 'categories'
    exists = 'exists'
    traditional = 'traditional'
    fuzzy = 'fuzzy'
    regexp = 'regexp'
    ids = 'ids'
    date = 'date'
    numeric = 'numeric'
    search = 'search'
    or_ = 'or'
    word_count = 'word_count'
    character_count = 'character_count'
    dedupe_by_value = 'dedupe_by_value'
    match_array = 'match_array'
    random = 'random'
    and_ = 'and'
    size = 'size'


class Filter(BaseModel):
    strict: Optional[Strict] = None
    condition: Optional[str] = None
    case_insensitive: Optional[bool] = None
    field: Optional[str] = None
    filter_type: Optional[FilterType] = None
    condition_value: Optional[Any] = None
    fuzzy: Optional[float] = None
    join: Optional[bool] = None

class Type2(Enum):
    email_read_write = 'email-read-write'
    calendar_read_write = 'calendar-read-write'
    microsoft_teams = 'microsoft-teams'
    meeting_read_write = 'meeting-read-write'
    salesforce_api = 'salesforce-api'
    slack_channel_post = 'slack-channel-post'
    slack_channel_post_read = 'slack-channel-post-read'
    zendesk_create_ticket = 'zendesk-create-ticket'
    hubspot_connect_app = 'hubspot-connect-app'
    linear_ticket_create = 'linear-ticket-create'
    unipile_linkedin = 'unipile-linkedin'
    unipile_whatsapp = 'unipile-whatsapp'
    outreach_api = 'outreach-api'
    zoom_api_v1 = 'zoom-api-v1'
    zoho_crm = 'zoho-crm'


class OauthPermission(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    provider: Provider
    types: List[Type2]


class OauthAccountProvider(Enum):
    google = 'google'
    microsoft = 'microsoft'
    salesforce = 'salesforce'
    slack = 'slack'
    zendesk = 'zendesk'
    hubspot = 'hubspot'
    linear = 'linear'
    outreach = 'outreach'
    zoom = 'zoom'
    unipile_linkedin = 'unipile_linkedin'
    unipile_whatsapp = 'unipile_whatsapp'
    zoho_crm = 'zoho_crm'


class OauthAccountPermissionType(Enum):
    email_read_write = 'email-read-write'
    calendar_read_write = 'calendar-read-write'
    microsoft_teams = 'microsoft-teams'
    meeting_read_write = 'meeting-read-write'
    salesforce_api = 'salesforce-api'
    slack_channel_post = 'slack-channel-post'
    slack_channel_post_read = 'slack-channel-post-read'
    zendesk_create_ticket = 'zendesk-create-ticket'
    hubspot_connect_app = 'hubspot-connect-app'
    linear_ticket_create = 'linear-ticket-create'
    unipile_linkedin = 'unipile-linkedin'
    unipile_whatsapp = 'unipile-whatsapp'
    outreach_api = 'outreach-api'
    zoom_api_v1 = 'zoom-api-v1'
    zoho_crm = 'zoho-crm'


class Type3(Enum):
    dynamic = 'dynamic'
    static = 'static'


class Scratchpad(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Type3


class Metadata2(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[ApiSelectorType] = None
    api_selector_placeholder: Optional[str] = 'Select Option...'
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = False
    relevance_only: Optional[bool] = False
    advanced: Optional[bool] = False
    placeholder: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = False
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[ValueSuggestionChain] = None
    enum: Optional[List[EnumItem]] = None
    bulk_run_input_source: Optional[BulkRunInputSource] = None
    agent_input_source: Optional[AgentInputSource] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[Dict[str, str]] = None
    language: Optional[Language] = None
    key_value_input_opts: Optional[KeyValueInputOpts] = Field(
        None, description='Props to pass to the KeyValueInput component.'
    )
    knowledge_set_field_name: Optional[str] = Field(
        None,
        description="[KnowledgeEditor] The name of the field in the transformation's param schema containing the knowledge set ID.",
    )
    filters: Optional[List[Filter]] = Field(
        None, description='General filters for the content_type'
    )
    oauth_permissions: Optional[List[OauthPermission]] = Field(
        None,
        description='(Optional) OAuth permissions required for a step. Only applicable for content_type `oauth_token`',
    )
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )
    oauth_account_provider: Optional[OauthAccountProvider] = Field(
        None,
        description='Filters the OAuth account selector based on the selected provider',
    )
    oauth_account_permission_type: Optional[OauthAccountPermissionType] = Field(
        None,
        description='Filters the OAuth account selector based on the selected permission type',
    )
    scratchpad: Optional[Scratchpad] = None


class Items(BaseModel):
    type: Optional[str] = None


class Properties(BaseModel):
    metadata: Optional[Metadata2] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema(BaseModel):
    metadata: Optional[Metadata1] = None
    properties: Optional[Dict[str, Properties]] = None


class Agent(BaseModel):
    message_template: Optional[str] = None


class AfterRetriesBehaviour(Enum):
    terminate_conversation = 'terminate-conversation'
    ask_for_approval = 'ask-for-approval'


class ActionRetryConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_retries: Optional[float] = None
    force_retry: Optional[bool] = None
    after_retries_behaviour: Optional[AfterRetriesBehaviour] = None


class ConditionalApprovalRules(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_auto_approvals: Optional[float] = None
    max_approvals_asked: Optional[float] = None


class DefaultValues(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    field_subagent_params: Optional[Dict[str, Any]] = Field(
        None,
        alias='_subagent_params',
        description='Params to substitute in the subagent. This is only used for subagents (when agent_id is set). This should be used instead of providing params at the top-level since that is reserved for the params in the message template defined in the parent agent.',
    )


class Action(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    chain_id: Optional[str] = None
    agent_id: Optional[str] = None
    params_schema: Optional[ParamsSchema] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    agent: Optional[Agent] = None
    action_behaviour: Optional[str] = None
    action_retry_config: Optional[ActionRetryConfig] = None
    title: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    agent_decide_prompt: Optional[str] = Field(
        None,
        description="This prompt guides the agent's decision on whether or not approval is required to execute the tool.",
    )
    conditional_approval_rules: Optional[ConditionalApprovalRules] = None
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description='Useful for external systems to track which tools have been added programatically.',
    )
    project: Optional[str] = Field(
        None,
        description='Defaults to project agent is being ran from. Tools must be public to use tools in other projects.',
    )
    region: Optional[str] = Field(
        None,
        description='Defaults to region agent is being ran from. Tools must be public to use tools in other regions.',
    )
    version: Optional[str] = Field(
        None,
        description="Version of the tool or subagent to run. Defaults to 'latest'.",
    )
    default_values: Optional[DefaultValues] = Field(
        None,
        description='Default values the agent will use as inputs when running the tool.',
    )
    auth_values: Optional[Dict[str, Any]] = Field(
        None,
        description='Auth account values to update in the chains. Will not be saved to the agent config.',
    )
    prompt_description: Optional[str] = None


class ActionRetryConfig1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_retries: Optional[float] = None
    force_retry: Optional[bool] = None
    after_retries_behaviour: Optional[AfterRetriesBehaviour] = None


class ImportanceLevel(Enum):
    normal = 'normal'
    short_term_memory = 'short-term-memory'


class Message(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    role: Literal['user']
    content: str
    importance_level: Optional[ImportanceLevel] = None


class Message1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    role: Literal['agent']
    content: str


class Metadata3(BaseModel):
    sync_item_id: Optional[str] = None
    sync_id: Optional[str] = None
    sync_type: Optional[str] = None


class StartingMessage(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    message: Union[Message, Message1]
    metadata: Optional[Metadata3] = Field(
        None,
        description='Any additional metadata to be stored with the message. This is not sent to the agent.',
    )


class Studio(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    studio_id: str
    label: Optional[str] = None


class Trigger(BaseModel):
    type: str
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    studios: Optional[List[Studio]] = None


class Runner(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Literal['default']


class ChildAgent(BaseModel):
    agent_id: Optional[str] = None
    system_prompt: Optional[str] = None
    name: Optional[str] = None


class Multiagent(BaseModel):
    max_rounds: Optional[float] = 5
    child_agents: Optional[List[ChildAgent]] = Field(
        None, description='Agents that the Admin agent will run.'
    )


class Runner1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Literal['multiagent']
    multiagent: Optional[Multiagent] = None


class Tags(BaseModel):
    description: Optional[str] = None


class InternalTags(BaseModel):
    priority: Optional[bool] = None


class Metadata4(BaseModel):
    field_order: Optional[List[str]] = None


class KeyValueInputOpts1(BaseModel):
    header: Optional[Header] = Field(
        None, description='Set headers to display above the key and/or value columns.'
    )
    placeholder: Optional[Placeholder] = Field(
        None,
        description='Set placeholder values to display in the key and/or value columns.',
    )
    addButtonText: Optional[str] = Field(
        None,
        description="The text displayed in the 'Add' button that inserts a new pair.",
    )


class Filter1(BaseModel):
    strict: Optional[Strict] = None
    condition: Optional[str] = None
    case_insensitive: Optional[bool] = None
    field: Optional[str] = None
    filter_type: Optional[FilterType] = None
    condition_value: Optional[Any] = None
    fuzzy: Optional[float] = None
    join: Optional[bool] = None


class Metadata5(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[ApiSelectorType] = None
    api_selector_placeholder: Optional[str] = 'Select Option...'
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = False
    relevance_only: Optional[bool] = False
    advanced: Optional[bool] = False
    placeholder: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = False
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[ValueSuggestionChain] = None
    enum: Optional[List[EnumItem]] = None
    bulk_run_input_source: Optional[BulkRunInputSource] = None
    agent_input_source: Optional[AgentInputSource] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[Dict[str, str]] = None
    language: Optional[Language] = None
    key_value_input_opts: Optional[KeyValueInputOpts1] = Field(
        None, description='Props to pass to the KeyValueInput component.'
    )
    knowledge_set_field_name: Optional[str] = Field(
        None,
        description="[KnowledgeEditor] The name of the field in the transformation's param schema containing the knowledge set ID.",
    )
    filters: Optional[List[Filter1]] = Field(
        None, description='General filters for the content_type'
    )
    oauth_permissions: Optional[List[OauthPermission]] = Field(
        None,
        description='(Optional) OAuth permissions required for a step. Only applicable for content_type `oauth_token`',
    )
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )
    oauth_account_provider: Optional[OauthAccountProvider] = Field(
        None,
        description='Filters the OAuth account selector based on the selected provider',
    )
    oauth_account_permission_type: Optional[OauthAccountPermissionType] = Field(
        None,
        description='Filters the OAuth account selector based on the selected permission type',
    )
    scratchpad: Optional[Scratchpad] = None


class Properties1(BaseModel):
    metadata: Optional[Metadata5] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema1(BaseModel):
    metadata: Optional[Metadata4] = None
    properties: Optional[Dict[str, Properties1]] = None


class Email(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    emails: Optional[Any] = None


class Channel(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    oauth_account_id: Optional[Any] = None


class Slack(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    channels: Optional[List[Channel]] = None


class Escalations(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    email: Optional[Email] = None
    slack: Optional[Slack] = None


class SwitchModelAfterNTokens(BaseModel):
    model: str
    n_tokens: float


class ModelOptions(BaseModel):
    parallel_tool_calls: Optional[bool] = None
    switch_model_after_n_tokens: Optional[SwitchModelAfterNTokens] = Field(
        None,
        description='Cost reduction technique due to models performing better with more context',
    )
    strict_mode: Optional[bool] = Field(
        None,
        description="Some model providers support 'strict':true to force function calling to be more accurate. This activates this when supported.",
    )


class Runtime(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    code: Optional[str] = None
    enabled: Optional[bool] = False


class LastUpdatedBy(BaseModel):
    user_name: Optional[str] = None
    user_id: Optional[str] = None


class Metadata6(BaseModel):
    clone_count: Optional[float] = None


class Agents(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    field_id: str = Field(..., alias='_id')
    agent_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    prompt_description: Optional[str] = None
    user_instructions: Optional[str] = Field(
        None,
        description='Used to provide instructions on how to use or set up the agent.',
    )
    template: Optional[Template] = Field(
        None,
        description='If set, the agents config will be completely replaced with the templates config.',
    )
    origin: Optional[Origin] = Field(
        None, description='If set, records where the agent was cloned from.'
    )
    emoji: Optional[str] = None
    views: Optional[float] = None
    max_job_duration: Optional[MaxJobDuration] = Field(
        None,
        description='Switching this to hours tells our runner engine to run the job in a way suited for long runs.',
    )
    title_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    suggest_replies: Optional[bool] = Field(
        False,
        description='If true, suggested replies will appear for each agent message in the front-end.',
    )
    suggest_replies_prompt: Optional[str] = Field(
        None,
        description='The prompt to inject into the LLM step in the tool used to generate suggested replies.',
    )
    is_scheduled_triggers_enabled: Optional[bool] = Field(
        False, description='If true, this agent can plan and schedule future actions'
    )
    model: Optional[str] = 'openai-gpt-4o-mini'
    autonomy_limit: Optional[float] = Field(
        10,
        description='Maximum number of actions an agent can autonomously take before the user needs to confirm again.',
    )
    temperature: Optional[float] = Field(
        0,
        description='Temperature of the selected model. Typically, higher temperature means more random output.',
    )
    knowledge: Optional[List[KnowledgeItem]] = None
    actions: Optional[List[Action]] = None
    action_behaviour: Optional[str] = 'always-ask'
    action_retry_config: Optional[ActionRetryConfig1] = None
    agent_decide_prompt: Optional[str] = Field(
        None,
        description="This prompt guides the agent's decision on whether or not approval is required to execute the tool.",
    )
    conditional_approval_rules: Optional[ConditionalApprovalRules] = None
    public: Optional[bool] = None
    in_marketplace: Optional[bool] = None
    project: str
    update_date_: Optional[str] = None
    version: Optional[str] = None
    embed_config: Optional[Dict[str, Any]] = None
    embeddable: Optional[bool] = None
    machine_user_id: Optional[str] = None
    starting_messages: Optional[List[StartingMessage]] = None
    triggers: Optional[List[Trigger]] = Field(
        None,
        description='Triggers are used to start / continue a conversation with an agent via an external service (e.g. email).',
    )
    runner: Optional[Union[Runner, Runner1]] = None
    tags: Optional[Dict[str, Union[bool, Tags]]] = None
    internal_tags: Optional[InternalTags] = None
    params_schema: Optional[ParamsSchema1] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    params: Optional[Dict[str, Any]] = None
    expiry_date_: Optional[Any] = None
    escalations: Optional[Escalations] = None
    use_streaming: Optional[bool] = Field(
        None,
        description="If true, the agent's progress will be streamed in real-time-ish to the frontend.",
    )
    mas_id: Optional[str] = Field(
        None,
        description='Can be used to force a given mas id for every run with this agent.',
    )
    model_options: Optional[ModelOptions] = None
    runtime: Optional[Runtime] = Field(
        None, description='Options for controlling of the agent runtime layer.'
    )
    categories: Optional[List[str]] = None
    last_updated_by: Optional[LastUpdatedBy] = None
    metadata: Optional[Metadata6] = None


### Task Conversation Summary

# generated by datamodel-codegen:
#   filename:  output_body_schema.json
#   timestamp: 2024-10-03T07:12:58+00:00


class Status(Enum):
    complete = 'complete'
    inprogress = 'inprogress'
    failed = 'failed'
    cancelled = 'cancelled'


class Error(BaseModel):
    body: Optional[str] = None


class Type(Enum):
    run_chain = 'run_chain'
    notebook = 'notebook'
    trigger_limited = 'trigger_limited'
    cron = 'cron'
    bulk_run = 'bulk_run'
    agent = 'agent'
    email = 'email'
    sync = 'sync'
    custom_gpt = 'custom_gpt'


class Executor(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    type: Type
    api_key_id: Optional[str] = None
    workflow_id: Optional[str] = None
    document_id: Optional[str] = None
    sync_id: Optional[str] = None
    job_id: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    email_message_id: Optional[str] = None


class CreditsUsedItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    credits: float
    name: str
    num_units: Optional[float] = None
    multiplier: Optional[float] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_run_id: Optional[str] = None


class MaxJobDuration(Enum):
    hours = 'hours'
    minutes = 'minutes'
    synchronous_seconds = 'synchronous_seconds'
    background_seconds = 'background_seconds'


class TaskConversation(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    field_id: Optional[str] = Field(None, alias='_id')
    version: Optional[str] = None
    studio_id: Optional[str] = None
    title: Optional[str] = None
    insert_date_: Optional[str] = None
    expiry_date_: Optional[Any] = None
    status: Optional[Status] = Field(
        None,
        description='Status of the workflow. Used for knowing when to send an email notification.',
    )
    errors: Optional[List[Error]] = None
    execution_time: Optional[float] = None
    project: Optional[str] = None
    executor: Optional[Executor] = None
    credits_used: Optional[List[CreditsUsedItem]] = None
    max_job_duration: Optional[MaxJobDuration] = Field(
        None,
        description='Switching this to hours tells our runner engine to run the job in a way suited for long runs.',
    )
    output_preview: Optional[Dict[str, Any]] = None
    input_params: Optional[Dict[str, Any]] = None
    cost: Optional[float] = None
    internal_job_id: Optional[str] = None
    internal_log_info: Optional[Dict[str, Any]] = None


class DefaultOutputValue(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    original_key: str
    updated_key: Optional[str] = None
    value: Any


class Step(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    name: str
    transformation: str
    params: Dict[str, Any]
    saved_params: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    default_output_values: Optional[List[DefaultOutputValue]] = None
    continue_on_error: Optional[bool] = None
    use_fallback_on_skip: Optional[bool] = None
    foreach: Optional[Union[str, List]] = None
    if_: Optional[Union[str, bool]] = Field(None, alias='if')
    display_name: Optional[str] = None


class Transformations(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    steps: List[Step]
    output: Optional[Dict[str, str]] = None


class HostType(Enum):
    batch = 'batch'
    lambda_ = 'lambda'
    instant_workflows = 'instant-workflows'
    none = 'none'


class Properties(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    params: Dict[str, Any]
    workflow_id: str
    host_type: Optional[HostType] = 'batch'
    dataset_id: Optional[str] = None
    version: Optional[str] = None


class Transformations1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    properties: Properties
    depends_on: Optional[List[str]] = None
    repeat: Optional[str] = Field(
        None,
        description='This should evaluate to a number. For example, it would be ${{ params.cluster_job_count }} or ${{ params.cluster_sizes.length }}. it will run a duplicate version of the workflow with a variable ${{ repeat_index }} that marks what index it belongs to.',
    )
    repeat_index: Optional[float] = None
    if_: Optional[str] = Field(None, alias='if')
    output_key: Optional[str] = Field(
        None,
        description="If this step outputs to status, its output will be accessible at output[output_key] in the parent job's status.",
    )
    passthrough_email: Optional[bool] = Field(
        None,
        description='whether the email config of the step should be applied to the parent',
    )


class Template(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    transformations: Dict[str, Transformations1]


class Type1(Enum):
    transformation = 'transformation'


class Categories(Enum):
    boolean_True = True


class Tags(BaseModel):
    type: Optional[Type1] = None
    categories: Optional[Dict[str, Categories]] = None
    integration_source: Optional[str] = Field(
        None,
        description="The source of the integration. For example, 'Knowledge: Linear', which imports data from Linear, this would be 'linear'.",
    )


class Metadata(BaseModel):
    field_order: Optional[List[str]] = None


class ContentType(Enum):
    json = 'json'
    json_list = 'json_list'
    long_text = 'long_text'
    short_text = 'short_text'
    file_url = 'file_url'
    file_urls = 'file_urls'
    llm_prompt = 'llm_prompt'
    speech = 'speech'
    code = 'code'
    dataset_id = 'dataset_id'
    knowledge_set = 'knowledge_set'
    markdown = 'markdown'
    chain_id = 'chain_id'
    chain_params = 'chain_params'
    file_to_text = 'file_to_text'
    file_to_text_llm_friendly = 'file_to_text_llm_friendly'
    memory_optimizer = 'memory_optimizer'
    memory = 'memory'
    table = 'table'
    agent_id = 'agent_id'
    api_key = 'api_key'
    key_value_input = 'key_value_input'
    knowledge_editor = 'knowledge_editor'
    oauth_account = 'oauth_account'
    datetime = 'datetime'
    api_selector = 'api_selector'
    llm_model_selector = 'llm_model_selector'
    colour_picker = 'colour_picker'
    conditional = 'conditional'
    external_field = 'external_field'
    tool_approval = 'tool_approval'


class ApiSelectorType(Enum):
    finetuning_model_select = 'finetuning_model_select'
    elevenlabs_voice_selector = 'elevenlabs_voice_selector'
    elevenlabs_model_selector = 'elevenlabs_model_selector'
    heygen_voice_selector = 'heygen_voice_selector'
    heygen_avatar_selector = 'heygen_avatar_selector'
    llm_model_selector = 'llm_model_selector'
    blandai_voice_selector = 'blandai_voice_selector'
    d_id_voice_selector = 'd_id_voice_selector'
    d_id_avatar_selector = 'd_id_avatar_selector'
    vapi_custom_phone_number_selector = 'vapi_custom_phone_number_selector'
    vapi_custom_assistant_selector = 'vapi_custom_assistant_selector'
    webflow_collections = 'webflow_collections'


class ValueSuggestionChain(BaseModel):
    url: str
    project_id: str
    output_key: Optional[str] = 'value'

class BulkRunInputSource(Enum):
    field_ = ''
    field_DOCUMENT = '$DOCUMENT'
    field_FIELD_PARAM_MAPPING = '$FIELD_PARAM_MAPPING'


class AgentInputSource(Enum):
    conversation_id = 'conversation_id'
    agent_id = 'agent_id'


class Language(Enum):
    python = 'python'
    javascript = 'javascript'
    html = 'html'


class Header(BaseModel):
    hide: Optional[bool] = Field(None, description='Whether to hide all headers.')
    key: Optional[str] = Field(
        None, description='The header displayed above the key column.'
    )
    value: Optional[str] = Field(
        None, description='The header displayed above the value column.'
    )


class Placeholder(BaseModel):
    hide: Optional[bool] = Field(None, description='Whether to hide all placeholders.')
    key: Optional[str] = Field(
        None, description='The placeholder to display in each cell of the key column.'
    )
    value: Optional[str] = Field(
        None, description='The placeholder to display in each cell of the value column.'
    )


class KeyValueInputOpts(BaseModel):
    header: Optional[Header] = Field(
        None, description='Set headers to display above the key and/or value columns.'
    )
    placeholder: Optional[Placeholder] = Field(
        None,
        description='Set placeholder values to display in the key and/or value columns.',
    )
    addButtonText: Optional[str] = Field(
        None,
        description="The text displayed in the 'Add' button that inserts a new pair.",
    )


class Strict(Enum):
    must = 'must'
    should = 'should'
    must_or = 'must_or'


class FilterType(Enum):
    text_match = 'text_match'
    word_match = 'word_match'
    term = 'term'
    terms = 'terms'
    text = 'text'
    texts = 'texts'
    match = 'match'
    contains = 'contains'
    substring = 'substring'
    class_ = 'class'
    category = 'category'
    exact_match = 'exact_match'
    classes = 'classes'
    categories = 'categories'
    exists = 'exists'
    traditional = 'traditional'
    fuzzy = 'fuzzy'
    regexp = 'regexp'
    ids = 'ids'
    date = 'date'
    numeric = 'numeric'
    search = 'search'
    or_ = 'or'
    word_count = 'word_count'
    character_count = 'character_count'
    dedupe_by_value = 'dedupe_by_value'
    match_array = 'match_array'
    random = 'random'
    and_ = 'and'
    size = 'size'


class Filter(BaseModel):
    strict: Optional[Strict] = None
    condition: Optional[str] = None
    case_insensitive: Optional[bool] = None
    field: Optional[str] = None
    filter_type: Optional[FilterType] = None
    condition_value: Optional[Any] = None
    fuzzy: Optional[float] = None
    join: Optional[bool] = None


class Provider(Enum):
    google = 'google'
    microsoft = 'microsoft'
    salesforce = 'salesforce'
    slack = 'slack'
    zendesk = 'zendesk'
    hubspot = 'hubspot'
    linear = 'linear'
    outreach = 'outreach'
    zoom = 'zoom'
    unipile_linkedin = 'unipile_linkedin'
    unipile_whatsapp = 'unipile_whatsapp'
    zoho_crm = 'zoho_crm'

class OauthPermission(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    provider: Provider
    types: List[Type2]


class Metadata1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[ApiSelectorType] = None
    api_selector_placeholder: Optional[str] = 'Select Option...'
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = False
    relevance_only: Optional[bool] = False
    advanced: Optional[bool] = False
    placeholder: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = False
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[ValueSuggestionChain] = None
    enum: Optional[List[EnumItem]] = None
    bulk_run_input_source: Optional[BulkRunInputSource] = None
    agent_input_source: Optional[AgentInputSource] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[Dict[str, str]] = None
    language: Optional[Language] = None
    key_value_input_opts: Optional[KeyValueInputOpts] = Field(
        None, description='Props to pass to the KeyValueInput component.'
    )
    knowledge_set_field_name: Optional[str] = Field(
        None,
        description="[KnowledgeEditor] The name of the field in the transformation's param schema containing the knowledge set ID.",
    )
    filters: Optional[List[Filter]] = Field(
        None, description='General filters for the content_type'
    )
    oauth_permissions: Optional[List[OauthPermission]] = Field(
        None,
        description='(Optional) OAuth permissions required for a step. Only applicable for content_type `oauth_token`',
    )
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )
    oauth_account_provider: Optional[OauthAccountProvider] = Field(
        None,
        description='Filters the OAuth account selector based on the selected provider',
    )
    oauth_account_permission_type: Optional[OauthAccountPermissionType] = Field(
        None,
        description='Filters the OAuth account selector based on the selected permission type',
    )
    scratchpad: Optional[Scratchpad] = None


class Items(BaseModel):
    type: Optional[str] = None


class Properties1(BaseModel):
    metadata: Optional[Metadata1] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema(BaseModel):
    metadata: Optional[Metadata] = None
    properties: Optional[Dict[str, Properties1]] = None


class Metadata2(BaseModel):
    field_order: Optional[List[str]] = Field(
        None,
        description='An array of output keys in the order that they should be displayed in the tool builder. Used in the frontend to guarantee tab order.',
    )


class ContentType1(Enum):
    html = 'html'
    chart_js = 'chart.js'
    external_field = 'external_field'


class Metadata3(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType1] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )


class Properties2(BaseModel):
    metadata: Optional[Metadata3] = None


class OutputSchema(BaseModel):
    metadata: Optional[Metadata2] = None
    properties: Optional[Dict[str, Properties2]] = None


class PredictedOutputItem(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    type: Optional[str] = None


class Frequency(Enum):
    hourly = 'hourly'
    daily = 'daily'
    every_2_minutes = 'every-2-minutes'


class Schedule(BaseModel):
    frequency: Optional[Frequency] = None


class Steps(BaseModel):
    output: Optional[Dict[str, Any]] = None
    executionTime: Optional[float] = None
    results: Optional[List] = Field(
        None,
        description='Only used if the corresponding step has a `foreach` configured',
    )
    skipped: Optional[bool] = Field(
        None, description='Will be true if the step was skipped, and not run'
    )
    skippedItems: Optional[List] = None


class State(BaseModel):
    params: Optional[Dict[str, Any]] = None
    steps: Optional[Dict[str, Steps]] = None


class Metadata4(BaseModel):
    source_studio_id: Optional[str] = None
    source_region: Optional[str] = None
    source_project: Optional[str] = None
    clone_count: Optional[float] = None


class Metrics(BaseModel):
    views: Optional[float] = None
    executions: Optional[float] = None


class Studios(BaseModel):
    version: Optional[str] = None
    project: Optional[str] = None
    field_id: Optional[str] = Field(None, alias='_id')
    studio_id: str
    public: Optional[bool] = Field(
        False, description='Anyone can view or clone this tool'
    )
    in_marketplace: Optional[bool] = Field(
        False, description='This tool is listed on the tool marketplace'
    )
    insert_date_: Optional[str] = None
    transformations: Optional[Transformations] = None
    template: Optional[Template] = Field(
        None,
        description='Templates support simple expressions on \'if\' and \'params\' fields. Supply an expression in the format: \n\n${{params.my_param + params.my_param2 <= 5 }}\n\nOperators supported: ["!=","==","+","-","&&","||","false","true"]\nEnsure that there is a space between each expression item or it will not evaluate correctly.',
    )
    update_date_: Optional[str] = None
    is_hidden: Optional[bool] = False
    tags: Optional[Tags] = None
    publicly_triggerable: Optional[bool] = Field(
        False, description='Anyone can run this tool'
    )
    machine_user_id: Optional[str] = None
    creator_user_id: Optional[str] = None
    creator_first_name: Optional[str] = None
    creator_last_name: Optional[str] = None
    creator_display_picture: Optional[str] = None
    cover_image: Optional[str] = None
    emoji: Optional[str] = None
    params_schema: Optional[ParamsSchema] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    output_schema: Optional[OutputSchema] = Field(
        None,
        description='A jsonschema superset object to provide metadata for tool output fields.',
    )
    predicted_output: Optional[List[PredictedOutputItem]] = None
    schedule: Optional[Schedule] = None
    state: Optional[State] = Field(
        None, description='Override the starting state of the studio'
    )
    title: Optional[str] = None
    description: Optional[str] = None
    prompt_description: Optional[str] = None
    state_mapping: Optional[Dict[str, str]] = Field(
        None, description='Mapping from alias -> real variable path'
    )
    max_job_duration: Optional[MaxJobDuration] = Field(
        None,
        description='Switching this to hours tells our runner engine to run the job in a way suited for long runs.',
    )
    metadata: Optional[Metadata4] = None
    metrics: Optional[Metrics] = None
    share_id: Optional[str] = None


class Region(Enum):
    field_1e3042 = '1e3042'
    f1db6c = 'f1db6c'
    d7b62b = 'd7b62b'
    bcbe5a = 'bcbe5a'
    field_ = ''


class Template1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    agent_id: str
    region: Optional[Region] = None
    project: Optional[str] = None


class Origin(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    root_agent_id: str
    region: Optional[str] = None
    project: Optional[str] = None


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    knowledge_set: str


class Metadata5(BaseModel):
    field_order: Optional[List[str]] = None


class ContentType2(Enum):
    json = 'json'
    json_list = 'json_list'
    long_text = 'long_text'
    short_text = 'short_text'
    file_url = 'file_url'
    file_urls = 'file_urls'
    llm_prompt = 'llm_prompt'
    speech = 'speech'
    code = 'code'
    dataset_id = 'dataset_id'
    knowledge_set = 'knowledge_set'
    markdown = 'markdown'
    chain_id = 'chain_id'
    chain_params = 'chain_params'
    file_to_text = 'file_to_text'
    file_to_text_llm_friendly = 'file_to_text_llm_friendly'
    memory_optimizer = 'memory_optimizer'
    memory = 'memory'
    table = 'table'
    agent_id = 'agent_id'
    api_key = 'api_key'
    key_value_input = 'key_value_input'
    knowledge_editor = 'knowledge_editor'
    oauth_account = 'oauth_account'
    datetime = 'datetime'
    api_selector = 'api_selector'
    llm_model_selector = 'llm_model_selector'
    colour_picker = 'colour_picker'
    conditional = 'conditional'
    external_field = 'external_field'
    tool_approval = 'tool_approval'


class KeyValueInputOpts1(BaseModel):
    header: Optional[Header] = Field(
        None, description='Set headers to display above the key and/or value columns.'
    )
    placeholder: Optional[Placeholder] = Field(
        None,
        description='Set placeholder values to display in the key and/or value columns.',
    )
    addButtonText: Optional[str] = Field(
        None,
        description="The text displayed in the 'Add' button that inserts a new pair.",
    )


class Filter1(BaseModel):
    strict: Optional[Strict] = None
    condition: Optional[str] = None
    case_insensitive: Optional[bool] = None
    field: Optional[str] = None
    filter_type: Optional[FilterType] = None
    condition_value: Optional[Any] = None
    fuzzy: Optional[float] = None
    join: Optional[bool] = None


class Metadata6(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType2] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[ApiSelectorType] = None
    api_selector_placeholder: Optional[str] = 'Select Option...'
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = False
    relevance_only: Optional[bool] = False
    advanced: Optional[bool] = False
    placeholder: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = False
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[ValueSuggestionChain] = None
    enum: Optional[List[EnumItem]] = None
    bulk_run_input_source: Optional[BulkRunInputSource] = None
    agent_input_source: Optional[AgentInputSource] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[Dict[str, str]] = None
    language: Optional[Language] = None
    key_value_input_opts: Optional[KeyValueInputOpts1] = Field(
        None, description='Props to pass to the KeyValueInput component.'
    )
    knowledge_set_field_name: Optional[str] = Field(
        None,
        description="[KnowledgeEditor] The name of the field in the transformation's param schema containing the knowledge set ID.",
    )
    filters: Optional[List[Filter1]] = Field(
        None, description='General filters for the content_type'
    )
    oauth_permissions: Optional[List[OauthPermission]] = Field(
        None,
        description='(Optional) OAuth permissions required for a step. Only applicable for content_type `oauth_token`',
    )
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )
    oauth_account_provider: Optional[OauthAccountProvider] = Field(
        None,
        description='Filters the OAuth account selector based on the selected provider',
    )
    oauth_account_permission_type: Optional[OauthAccountPermissionType] = Field(
        None,
        description='Filters the OAuth account selector based on the selected permission type',
    )
    scratchpad: Optional[Scratchpad] = None


class Properties3(BaseModel):
    metadata: Optional[Metadata6] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema1(BaseModel):
    metadata: Optional[Metadata5] = None
    properties: Optional[Dict[str, Properties3]] = None


class Agent(BaseModel):
    message_template: Optional[str] = None


class AfterRetriesBehaviour(Enum):
    terminate_conversation = 'terminate-conversation'
    ask_for_approval = 'ask-for-approval'


class ActionRetryConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_retries: Optional[float] = None
    force_retry: Optional[bool] = None
    after_retries_behaviour: Optional[AfterRetriesBehaviour] = None


class ConditionalApprovalRules(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_auto_approvals: Optional[float] = None
    max_approvals_asked: Optional[float] = None


class DefaultValues(BaseModel):
    model_config = ConfigDict(
        extra='allow',
    )
    field_subagent_params: Optional[Dict[str, Any]] = Field(
        None,
        alias='_subagent_params',
        description='Params to substitute in the subagent. This is only used for subagents (when agent_id is set). This should be used instead of providing params at the top-level since that is reserved for the params in the message template defined in the parent agent.',
    )


class Action(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    chain_id: Optional[str] = None
    agent_id: Optional[str] = None
    params_schema: Optional[ParamsSchema1] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    agent: Optional[Agent] = None
    action_behaviour: Optional[str] = None
    action_retry_config: Optional[ActionRetryConfig] = None
    title: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    agent_decide_prompt: Optional[str] = Field(
        None,
        description="This prompt guides the agent's decision on whether or not approval is required to execute the tool.",
    )
    conditional_approval_rules: Optional[ConditionalApprovalRules] = None
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description='Useful for external systems to track which tools have been added programatically.',
    )
    project: Optional[str] = Field(
        None,
        description='Defaults to project agent is being ran from. Tools must be public to use tools in other projects.',
    )
    region: Optional[str] = Field(
        None,
        description='Defaults to region agent is being ran from. Tools must be public to use tools in other regions.',
    )
    version: Optional[str] = Field(
        None,
        description="Version of the tool or subagent to run. Defaults to 'latest'.",
    )
    default_values: Optional[DefaultValues] = Field(
        None,
        description='Default values the agent will use as inputs when running the tool.',
    )
    auth_values: Optional[Dict[str, Any]] = Field(
        None,
        description='Auth account values to update in the chains. Will not be saved to the agent config.',
    )
    prompt_description: Optional[str] = None


class ActionRetryConfig1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    max_retries: Optional[float] = None
    force_retry: Optional[bool] = None
    after_retries_behaviour: Optional[AfterRetriesBehaviour] = None


class ImportanceLevel(Enum):
    normal = 'normal'
    short_term_memory = 'short-term-memory'

class Metadata7(BaseModel):
    sync_item_id: Optional[str] = None
    sync_id: Optional[str] = None
    sync_type: Optional[str] = None


class StartingMessage(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    message: Union[Message, Message1]
    metadata: Optional[Metadata7] = Field(
        None,
        description='Any additional metadata to be stored with the message. This is not sent to the agent.',
    )


class Studio(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    studio_id: str
    label: Optional[str] = None


class Trigger(BaseModel):
    type: str
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    studios: Optional[List[Studio]] = None


class Runner(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Literal['default']


class ChildAgent(BaseModel):
    agent_id: Optional[str] = None
    system_prompt: Optional[str] = None
    name: Optional[str] = None


class Multiagent(BaseModel):
    max_rounds: Optional[float] = 5
    child_agents: Optional[List[ChildAgent]] = Field(
        None, description='Agents that the Admin agent will run.'
    )


class Runner1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Literal['multiagent']
    multiagent: Optional[Multiagent] = None


class InternalTags(BaseModel):
    priority: Optional[bool] = None


class Metadata8(BaseModel):
    field_order: Optional[List[str]] = None


class KeyValueInputOpts2(BaseModel):
    header: Optional[Header] = Field(
        None, description='Set headers to display above the key and/or value columns.'
    )
    placeholder: Optional[Placeholder] = Field(
        None,
        description='Set placeholder values to display in the key and/or value columns.',
    )
    addButtonText: Optional[str] = Field(
        None,
        description="The text displayed in the 'Add' button that inserts a new pair.",
    )

class Filter2(BaseModel):
    strict: Optional[Strict] = None
    condition: Optional[str] = None
    case_insensitive: Optional[bool] = None
    field: Optional[str] = None
    filter_type: Optional[FilterType] = None
    condition_value: Optional[Any] = None
    fuzzy: Optional[float] = None
    join: Optional[bool] = None


class Metadata9(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    content_type: Optional[ContentType2] = None
    allow_one_of_variable_mode: Optional[bool] = None
    api_selector_type: Optional[ApiSelectorType] = None
    api_selector_placeholder: Optional[str] = 'Select Option...'
    variable_search_field: Optional[str] = None
    accepted_file_types: Optional[List[str]] = None
    hidden: Optional[bool] = False
    relevance_only: Optional[bool] = False
    advanced: Optional[bool] = False
    placeholder: Optional[Any] = None
    title: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    require_toggle: Optional[bool] = None
    dont_substitute: Optional[bool] = False
    min: Optional[float] = None
    max: Optional[float] = None
    value_suggestion_chain: Optional[ValueSuggestionChain] = None
    enum: Optional[List[EnumItem]] = None
    bulk_run_input_source: Optional[BulkRunInputSource] = None
    agent_input_source: Optional[AgentInputSource] = None
    headers: Optional[List[str]] = None
    rows: Optional[float] = None
    can_add_or_remove_columns: Optional[bool] = None
    placeholders: Optional[Dict[str, str]] = None
    language: Optional[Language] = None
    key_value_input_opts: Optional[KeyValueInputOpts2] = Field(
        None, description='Props to pass to the KeyValueInput component.'
    )
    knowledge_set_field_name: Optional[str] = Field(
        None,
        description="[KnowledgeEditor] The name of the field in the transformation's param schema containing the knowledge set ID.",
    )
    filters: Optional[List[Filter2]] = Field(
        None, description='General filters for the content_type'
    )
    oauth_permissions: Optional[List[OauthPermission]] = Field(
        None,
        description='(Optional) OAuth permissions required for a step. Only applicable for content_type `oauth_token`',
    )
    is_fixed_param: Optional[bool] = None
    is_history_excluded: Optional[bool] = None
    auto_stringify: Optional[bool] = None
    external_name: Optional[str] = Field(
        None,
        description="Field name in external data source (e.g. 'agent_id' in agent conversation metadata)",
    )
    oauth_account_provider: Optional[OauthAccountProvider] = Field(
        None,
        description='Filters the OAuth account selector based on the selected provider',
    )
    oauth_account_permission_type: Optional[OauthAccountPermissionType] = Field(
        None,
        description='Filters the OAuth account selector based on the selected permission type',
    )
    scratchpad: Optional[Scratchpad] = None


class Properties4(BaseModel):
    metadata: Optional[Metadata9] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema2(BaseModel):
    metadata: Optional[Metadata8] = None
    properties: Optional[Dict[str, Properties4]] = None


class Email(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    emails: Optional[Any] = None


class Channel(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    oauth_account_id: Optional[Any] = None


class Slack(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    channels: Optional[List[Channel]] = None


class Escalations(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    email: Optional[Email] = None
    slack: Optional[Slack] = None


class SwitchModelAfterNTokens(BaseModel):
    model: str
    n_tokens: float


class ModelOptions(BaseModel):
    parallel_tool_calls: Optional[bool] = None
    switch_model_after_n_tokens: Optional[SwitchModelAfterNTokens] = Field(
        None,
        description='Cost reduction technique due to models performing better with more context',
    )
    strict_mode: Optional[bool] = Field(
        None,
        description="Some model providers support 'strict':true to force function calling to be more accurate. This activates this when supported.",
    )


class Runtime(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    code: Optional[str] = None
    enabled: Optional[bool] = False


class LastUpdatedBy(BaseModel):
    user_name: Optional[str] = None
    user_id: Optional[str] = None


class Metadata10(BaseModel):
    clone_count: Optional[float] = None


class AgentDetails(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    field_id: str = Field(..., alias='_id')
    agent_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    prompt_description: Optional[str] = None
    user_instructions: Optional[str] = Field(
        None,
        description='Used to provide instructions on how to use or set up the agent.',
    )
    template: Optional[Template1] = Field(
        None,
        description='If set, the agents config will be completely replaced with the templates config.',
    )
    origin: Optional[Origin] = Field(
        None, description='If set, records where the agent was cloned from.'
    )
    emoji: Optional[str] = None
    views: Optional[float] = None
    max_job_duration: Optional[MaxJobDuration] = Field(
        None,
        description='Switching this to hours tells our runner engine to run the job in a way suited for long runs.',
    )
    title_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    suggest_replies: Optional[bool] = Field(
        False,
        description='If true, suggested replies will appear for each agent message in the front-end.',
    )
    suggest_replies_prompt: Optional[str] = Field(
        None,
        description='The prompt to inject into the LLM step in the tool used to generate suggested replies.',
    )
    is_scheduled_triggers_enabled: Optional[bool] = Field(
        False, description='If true, this agent can plan and schedule future actions'
    )
    model: Optional[str] = 'openai-gpt-4o-mini'
    autonomy_limit: Optional[float] = Field(
        10,
        description='Maximum number of actions an agent can autonomously take before the user needs to confirm again.',
    )
    temperature: Optional[float] = Field(
        0,
        description='Temperature of the selected model. Typically, higher temperature means more random output.',
    )
    knowledge: Optional[List[KnowledgeItem]] = None
    actions: Optional[List[Action]] = None
    action_behaviour: Optional[str] = 'always-ask'
    action_retry_config: Optional[ActionRetryConfig1] = None
    agent_decide_prompt: Optional[str] = Field(
        None,
        description="This prompt guides the agent's decision on whether or not approval is required to execute the tool.",
    )
    conditional_approval_rules: Optional[ConditionalApprovalRules] = None
    public: Optional[bool] = None
    in_marketplace: Optional[bool] = None
    project: str
    update_date_: Optional[str] = None
    version: Optional[str] = None
    embed_config: Optional[Dict[str, Any]] = None
    embeddable: Optional[bool] = None
    machine_user_id: Optional[str] = None
    starting_messages: Optional[List[StartingMessage]] = None
    triggers: Optional[List[Trigger]] = Field(
        None,
        description='Triggers are used to start / continue a conversation with an agent via an external service (e.g. email).',
    )
    runner: Optional[Union[Runner, Runner1]] = None
    tags: Optional[Dict[str, Union[bool, Tags]]] = None
    internal_tags: Optional[InternalTags] = None
    params_schema: Optional[ParamsSchema2] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    params: Optional[Dict[str, Any]] = None
    expiry_date_: Optional[Any] = None
    escalations: Optional[Escalations] = None
    use_streaming: Optional[bool] = Field(
        None,
        description="If true, the agent's progress will be streamed in real-time-ish to the frontend.",
    )
    mas_id: Optional[str] = Field(
        None,
        description='Can be used to force a given mas id for every run with this agent.',
    )
    model_options: Optional[ModelOptions] = None
    runtime: Optional[Runtime] = Field(
        None, description='Options for controlling of the agent runtime layer.'
    )
    categories: Optional[List[str]] = None
    last_updated_by: Optional[LastUpdatedBy] = None
    metadata: Optional[Metadata10] = None
    
### Triggered Task

class JobInfo(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    job_id: str
    studio_id: str


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


class TriggeredTask(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    job_info: JobInfo
    conversation_id: str
    agent_id: str
    state: State

    def __repr__(self):
        return f"TriggeredTask(conversation_id=\"{self.conversation_id}\")"
    
### Scheduled Action Trigger

class ScheduledActionTrigger(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    trigger_id: str

    def __repr__(self):
        return f"ScheduledActionTrigger(trigger_id=\"{self.trigger_id}\")"
    
### Task View


class Debug(BaseModel):
    original_documents: Optional[List[Dict[str, Any]]] = None


class Feedback(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    task_details: Optional[str] = None
    original_task: Optional[str] = None
    task_id: Optional[str] = None
    message_id: str
    task_tags: Optional[List[str]] = None
    notes: Optional[str] = None
    is_full_conversation_feedback: Optional[bool] = None
    is_feedback_positive: Optional[bool] = None
    user_name: Optional[str] = None


class Display(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: Optional[str] = None
    content: Optional[str] = None
    icon: Optional[str] = None


class CallerAgent(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    agent_id: Optional[str] = None
    project: Optional[str] = None
    region: Optional[str] = None
    conversation_id: Optional[str] = None


class OriginalMessageIds(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    user: str


class Content(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['user-message']
    text: str
    user_id: Optional[str] = None
    display: Optional[Display] = None
    caller_agent: Optional[CallerAgent] = None
    is_trigger_message: Optional[bool] = None
    original_message_ids: OriginalMessageIds


class OriginalMessageIds1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    agent: str


class Content1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['agent-message']
    text: str
    generating: Optional[bool] = None
    original_message_ids: OriginalMessageIds1


class ToolRunState(Enum):
    pending = 'pending'
    running = 'running'
    finished = 'finished'
    error = 'error'
    cancelled = 'cancelled'


class Type(Enum):
    agent = 'agent'
    tool = 'tool'


class ToolConfig(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Type
    title: str
    description: str
    region: Optional[str] = None
    project: Optional[str] = None
    id: str
    version: str
    emoji: Optional[str] = None
    params_schema: Dict[str, Any]


class ActionDetails(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    action_request_id: str
    action: str


class Display1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: Optional[str] = None
    icon: Optional[str] = None


class By(Enum):
    user = 'user'
    agent = 'agent'


class Confirmation(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    confirmed: bool
    by: By


class ParamSource(Enum):
    llm = 'llm'
    external_field = 'external-field'
    action_confirm_params_override = 'action-confirm-params-override'
    debug_mode_input_override = 'debug-mode-input-override'
    agent_action_config_default_value = 'agent-action-config-default-value'
    scratchpad = 'scratchpad'


class Metadata(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    param_source: Optional[ParamSource] = None


class Params(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    valid: Literal[True]
    json_: Dict[str, Any] = Field(..., alias='json')
    overrides: Optional[Dict[str, Any]] = None
    resolved: Optional[Dict[str, Any]] = Field(
        None, description='The params with which the tool was actually run'
    )
    metadata: Optional[Dict[str, Metadata]] = None


class Params1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    valid: Literal[False]
    json_: str = Field(..., alias='json')


class Error(BaseModel):
    body: Optional[str] = None


class OutputSource(Enum):
    tool_run = 'tool-run'
    debug_mode_output_override = 'debug-mode-output-override'
    action_confirm_mock_tool_output = 'action-confirm-mock-tool-output'


class Provider1(Enum):
    gmail = 'gmail'
    outlook = 'outlook'
    sendgrid = 'sendgrid'


class Options(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    to_email: Optional[Union[str, List[str]]] = None
    cc_emails: Optional[List[str]] = None
    bcc_emails: Optional[List[str]] = None
    from_email: Optional[str] = None
    email_subject: Optional[str] = None
    email_body: Optional[str] = None
    provider: Optional[Provider1] = None


class Component(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    content_type: Literal['email']
    options: Options


class Response(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    state: Literal['awaiting-response']


class Display2(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: Optional[str] = None
    content: Optional[str] = None
    icon: Optional[str] = None


class Response1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    state: Literal['responded']
    text: str
    user_id: Optional[str] = None
    display: Optional[Display2] = None
    insert_date_: str


class Response2(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    state: Literal['skipped']


class Options1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    context: str
    response: Union[Response, Response1, Response2]


class Component1(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    content_type: Literal['escalation']
    options: Options1


class OriginalMessageIds2(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    action_request: str = Field(..., alias='action-request')
    action_confirm: Optional[str] = Field(None, alias='action-confirm')
    action_reject: Optional[str] = Field(None, alias='action-reject')
    action_response: Optional[str] = Field(None, alias='action-response')
    action_error: Optional[str] = Field(None, alias='action-error')


class Content2(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['tool-run']
    thoughts: Optional[str] = None
    generating: Optional[bool] = None
    tool_run_state: ToolRunState
    tool_config: ToolConfig
    action_details: ActionDetails
    display: Optional[Display1] = None
    requires_confirmation: bool
    confirmation: Optional[Confirmation] = None
    params: Union[Params, Params1]
    errors: Optional[List[Error]] = None
    output: Optional[Dict[str, Any]] = None
    output_source: Optional[OutputSource] = None
    optimistic_output: Optional[Dict[str, Any]] = None
    component: Optional[Union[Component, Component1]] = None
    original_message_ids: OriginalMessageIds2


class OriginalMessageIds3(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    agent_error: str = Field(..., alias='agent-error')


class Content3(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    type: Literal['agent-error']
    errors: List[Error]
    original_message_ids: OriginalMessageIds3


class TaskStep(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    item_id: str
    insert_date_: str
    is_expanded_by_default: bool = Field(
        ...,
        description='Whether the UI item in the frontend is expanded (vs collapsed) by default',
    )
    is_in_hidden_group: bool
    debug: Optional[Debug] = None
    feedback: Optional[Feedback] = None
    content: Union[Content, Content1, Content2, Content3]

class TaskView(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    results: List[TaskStep] = Field(
        ..., description='The task view items for the frontend to render.'
    )
    next_cursor: Optional[str] = Field(
        None,
        description='The "after" cursor to use in the next request to get the next page of results. If one is not returned, there are no more results to fetch.',
    )