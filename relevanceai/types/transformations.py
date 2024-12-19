from __future__ import annotations
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TransformationBase(BaseModel):
    transformation: str
    name: str
    params: Dict[str, Any]
    output: Optional[Dict[str, str]] = None
    display_name: Optional[str] = None

class PromptCompletionTransformation(TransformationBase):
    transformation: str = "prompt_completion"
    params: Dict[str, Any] = Field(
        ..., 
        example={
            "prompt": "Tell me about AI Agents",
            "model": "openai-gpt4"
        }
    )

class PythonCodeTransformation(TransformationBase):
    transformation: str = "python_code_transformation"
    params: Dict[str, Any] = Field(
        ...,
        example={
            "code": "\nreturn \"Hello World!\""
        }
    )

class SerperGoogleSearchTransformation(TransformationBase):
    transformation: str = "serper_google_search"
    params: Dict[str, Any] = Field(
        ...,
        example={
            "search_query": "Relevance AI"
        }
    )

class TransformationStep(BaseModel):
    steps: List[TransformationBase]

    class Config:
        smart_union = True