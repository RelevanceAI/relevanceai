# generated by datamodel-codegen:
#   filename:  output_body_schema.json
#   timestamp: 2024-11-21T03:01:11+00:00

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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


class MessageAttachment(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: str
    url: str
    contentType: Optional[str] = None


class Display(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    name: Optional[str] = None
    content: Optional[str] = None
    content_html: Optional[str] = None
    message_attachments: Optional[List[MessageAttachment]] = None
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
    sync_type: Optional[str] = None
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


class Provider(Enum):
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
    provider: Optional[Provider] = None


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
    content_html: Optional[str] = None
    message_attachments: Optional[List[MessageAttachment]] = None
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


class Result(BaseModel):
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


class Model(BaseModel):
    model_config = ConfigDict(
        extra='forbid',
    )
    results: List[Result] = Field(
        ..., description='The task view items for the frontend to render.'
    )
    next_cursor: Optional[str] = Field(
        None,
        description='The "after" cursor to use in the next request to get the next page of results. If one is not returned, there are no more results to fetch.',
    )
