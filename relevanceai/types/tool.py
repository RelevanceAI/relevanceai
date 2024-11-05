from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, constr


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


class Template(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    transformations: Dict[str, Transformations]


class Type(Enum):
    transformation = 'transformation'


class Categories(Enum):
    boolean_True = True


class Tags(BaseModel):
    type: Optional[Type] = None
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


class Type1(Enum):
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
    types: List[Type1]


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


class Type2(Enum):
    dynamic = 'dynamic'
    static = 'static'


class Scratchpad(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    type: Type2


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


class MaxJobDuration(Enum):
    hours = 'hours'
    minutes = 'minutes'
    synchronous_seconds = 'synchronous_seconds'
    background_seconds = 'background_seconds'


class Metadata4(BaseModel):
    source_studio_id: Optional[str] = None
    source_region: Optional[str] = None
    source_project: Optional[str] = None
    clone_count: Optional[float] = None


class Metrics(BaseModel):
    views: Optional[float] = None
    executions: Optional[float] = None


class Tool(BaseModel):
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
    type: Optional[str] = None
    action_id: Optional[str] = None
    region: Optional[str] = None
    project: Optional[str] = None
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

    def __repr__(self):
        return f"Tool(tool_id=\"{self.studio_id}\", title=\"{self.title}\")"

### Tool Outputs 

class Status(Enum):
    complete = 'complete'
    inprogress = 'inprogress'
    failed = 'failed'
    cancelled = 'cancelled'


class Error(BaseModel):
    body: Optional[str] = None


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


class ToolOutput(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=() 
    )
    output: Dict[str, Any]
    state: Optional[Dict[str, Any]] = None
    status: Status = Field(
        ...,
        description='Status of the workflow. Used for knowing when to send an email notification.',
    )
    errors: List[Error]
    cost: Optional[float] = None
    credits_used: Optional[List[CreditsUsedItem]] = None
    executionTime: float

