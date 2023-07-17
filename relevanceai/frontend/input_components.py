"""
    All components are here
"""
from dataclasses import dataclass, field
from typing import List

class InputComponent:
    """
    Base class for all components
    """
    advanced:bool = False


@dataclass
class FileUpload(InputComponent):
    content_type: str = "file_url"
    accepted_file_type: list = field(default_factory=lambda: [])

@dataclass
class LongText(InputComponent):
    content_type: str = "long_text"

@dataclass
class Code(InputComponent):
    content_type: str = "code"

@dataclass
class LLMPrompt(InputComponent):
    content_type: str = "llm_prompt"

@dataclass
class Speech(InputComponent):
    content_type: str = "speech"