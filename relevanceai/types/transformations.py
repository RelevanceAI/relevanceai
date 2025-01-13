from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class TransformationBase(BaseModel):
    transformation: str
    name: str
    params: Dict[str, Any]
    output: Optional[Dict[str, str]] = None
    display_name: Optional[str] = None

class PromptCompletionTransformation(TransformationBase):
    transformation: str = "prompt_completion"
    name: str = "llm"
    params: Dict[str, Any] = Field(
        ..., 
        example={
            "prompt": "Tell me about AI Agents",
            "model": "openai-gpt4o"
        }
    )

class PythonCodeTransformation(TransformationBase):
    transformation: str = "python_code_transformation"
    name: str = "python"
    params: Dict[str, Any] = Field(
        ...,
        example={
            "code": "\nreturn \"Hello World!\""
        }
    )

class SerperGoogleSearchTransformation(TransformationBase):
    transformation: str = "serper_google_search"
    name: str = "google"
    params: Dict[str, Any] = Field(
        ...,
        example={
            "search_query": "Relevance AI"
        }
    )

class TransformationStep(BaseModel):
    steps: List[TransformationBase]

    model_config = ConfigDict(
        extra='allow',
        protected_namespaces=()
    )