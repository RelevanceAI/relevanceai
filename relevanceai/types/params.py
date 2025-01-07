from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class ParamsBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    order: int
    type: str
    value: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    default: Optional[Any] = None
    required: Optional[bool] = True

class TextParam(ParamsBase):
    type: str = "string"

class LongTextParam(ParamsBase):
    type: str = "string"
    metadata: Dict[str, str] = Field(default_factory=lambda: {"content_type": "long_text"})

class OptionsParam(ParamsBase):
    type: str = "string"
    enum: List[str]
    value: str

class NumberParam(ParamsBase):
    type: str = "number"
    min: Optional[float] = None
    max: Optional[float] = None
    value: float = 0

class CheckboxParam(ParamsBase):
    type: str = "boolean"
    value: bool = False
    default: bool = False

class ListParam(ParamsBase):
    type: str = "array"
    items: Dict[str, str] = Field(default_factory=lambda: {"type": "string"})
    value: List[str] = Field(default_factory=list)

class JsonParam(ParamsBase):
    type: str = "object"
    value: Dict[str, Any] = Field(default_factory=dict)

class JsonListParam(ParamsBase):
    type: str = "array"
    items: Dict[str, str] = Field(default_factory=lambda: {"type": "object"})
    value: List[Dict[str, Any]] = Field(default_factory=list)

class FileParam(ParamsBase):
    type: str = "string"
    value: str = ""
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "accepted_file_types": []
        }
    )

class FileTextParam(FileParam):
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "file_to_text_llm_friendly",
            "accepted_file_types": []
        }
    )

class FileUrlParam(FileParam):
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "file_url",
            "accepted_file_types": []
        }
    )

class FileUrlsParam(ParamsBase):
    type: str = "array"
    items: Dict[str, str] = Field(default_factory=lambda: {"type": "string"})
    value: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "file_urls",
            "accepted_file_types": []
        }
    )

class TableParam(ParamsBase):
    type: str = "array"
    items: Dict[str, str] = Field(default_factory=lambda: {"type": "object"})
    value: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "table",
            "headers": []
        }
    )

class ApiKeyParam(ParamsBase):
    type: str = "string"
    value: str = ""
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "api_key"
        }
    )

class KnowledgeTableParam(ParamsBase):
    type: str = "string"
    value: str = ""
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "knowledge_set"
        }
    )

class OAuthAccountParam(ParamsBase):
    type: str = "string"
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "content_type": "oauth_account"
        }
    )

class ParamsSchema(BaseModel):
    text: TextParam
    long_text: LongTextParam
    options: OptionsParam
    number: NumberParam
    checkbox: CheckboxParam
    list: ListParam
    json_object: JsonParam
    json_list: JsonListParam
    file_text: FileTextParam
    file_url: FileUrlParam
    file_urls: FileUrlsParam
    table: TableParam
    api_key: ApiKeyParam
    knowledge_table: KnowledgeTableParam
    oauth_account_id: OAuthAccountParam