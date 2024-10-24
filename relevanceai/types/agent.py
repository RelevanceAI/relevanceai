from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Region(Enum):
    field_1e3042 = '1e3042'
    f1db6c = 'f1db6c'
    d7b62b = 'd7b62b'
    bcbe5a = 'bcbe5a'
    field_ = ''


class Template(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    agent_id: str
    region: Optional[Region] = None
    project: Optional[str] = None


class Origin(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
        extra='allow',
        protected_namespaces=() 
    )
    knowledge_set: str

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
        extra='allow',
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


class Type(Enum):
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
        extra='allow',
        protected_namespaces=() 
    )
    provider: Provider
    types: List[Type]


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


class Type1(Enum):
    dynamic = 'dynamic'
    static = 'static'


class Scratchpad(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    type: Type1


class Metadata1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
    metadata: Optional[Metadata1] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class ParamsSchema(BaseModel):
    metadata: Optional[Metadata] = None
    properties: Optional[Dict[str, Properties]] = None


class ActionAgent(BaseModel):
    message_template: Optional[str] = None


class AfterRetriesBehaviour(Enum):
    terminate_conversation = 'terminate-conversation'
    ask_for_approval = 'ask-for-approval'


class ActionRetryConfig(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    max_retries: Optional[float] = None
    force_retry: Optional[bool] = None
    after_retries_behaviour: Optional[AfterRetriesBehaviour] = None


class ConditionalApprovalRules(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
        extra='allow',
        protected_namespaces=() 
    )
    chain_id: Optional[str] = None
    agent_id: Optional[str] = None
    params_schema: Optional[ParamsSchema] = Field(
        None,
        description='A jsonschema superset object that users parameters will be validated against upon execution.',
    )
    agent: Optional[ActionAgent] = None
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

class ImportanceLevel(Enum):
    normal = 'normal'
    short_term_memory = 'short-term-memory'


class Message(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    role: Literal['user']
    content: str
    importance_level: Optional[ImportanceLevel] = None


class Message1(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    role: Literal['agent']
    content: str


class Metadata2(BaseModel):
    sync_item_id: Optional[str] = None
    sync_id: Optional[str] = None
    sync_type: Optional[str] = None


class StartingMessage(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    message: Union[Message, Message1]
    metadata: Optional[Metadata2] = Field(
        None,
        description='Any additional metadata to be stored with the message. This is not sent to the agent.',
    )


class Studio(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
        extra='allow',
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
        extra='allow',
        protected_namespaces=() 
    )
    type: Literal['multiagent']
    multiagent: Optional[Multiagent] = None


class Tags(BaseModel):
    description: Optional[str] = None


class InternalTags(BaseModel):
    priority: Optional[bool] = None


class Metadata3(BaseModel):
    field_order: Optional[List[str]] = None

class Metadata4(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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


class Properties1(BaseModel):
    metadata: Optional[Metadata4] = None
    order: Optional[float] = None
    items: Optional[Items] = None


class Email(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    emails: Optional[Any] = None


class Channel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    oauth_account_id: Optional[Any] = None


class Slack(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=() 
    )
    channels: Optional[List[Channel]] = None


class Escalations(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
        extra='allow',
        protected_namespaces=() 
    )
    code: Optional[str] = None
    enabled: Optional[bool] = False


class LastUpdatedBy(BaseModel):
    user_name: Optional[str] = None
    user_id: Optional[str] = None


class Metadata5(BaseModel):
    clone_count: Optional[float] = None


class Agent(BaseModel):
    model_config = ConfigDict(
        extra='allow',
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
    action_retry_config: Optional[ActionRetryConfig] = None
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
    params_schema: Optional[ParamsSchema] = Field(
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
    metadata: Optional[Metadata5] = None

    def __repr__(self):
        return f"Agent(agent_id=\"{self.agent_id}\", name=\"{self.name}\")"

