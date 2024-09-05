
from pydantic import BaseModel
from typing import List, Optional

run = {
  "id": "run_abc123",
  "object": "task.run",
  "created_at": 1698107661,
  "assistant_id": "asst_abc123",
  "task_id": "task_abc123",
  "status": "completed",
  "started_at": 1699073476,
  "expires_at": None,
  "cancelled_at": None,
  "failed_at": None,
  "completed_at": 1699073498,
  "last_error": None,
  "model": "gpt-4o",
  "instructions": None,
  "tools": [{"type": "file_search"}, {"type": "code_interpreter"}],
  "metadata": {},
  "incomplete_details": None,
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "temperature": 1.0,
  "top_p": 1.0,
  "max_prompt_tokens": 1000,
  "max_completion_tokens": 1000,
  "truncation_strategy": {
    "type": "auto",
    "last_messages": None
  },
  "response_format": "auto",
  "tool_choice": "auto",
  "parallel_tool_calls": True
}

class Run(BaseModel):
    id: str
    object: str 
    created_at: int
    assistant_id: str
    task_id: str
    status: str
    started_at: Optional[int]
    expires_at: Optional[int]
    cancelled_at: Optional[int]
    failed_at: Optional[int]
    completed_at: Optional[int]
    last_error: Optional[str]
    model: Optional[str]
    instructions: Optional[str]
    tools: List[dict]
    metadata: dict
    incomplete_details: Optional[str]
    usage: dict
    temperature: Optional[float]
    top_p: Optional[float]
    max_prompt_tokens: Optional[int]
    max_completion_tokens: Optional[int]
    truncation_strategy: dict
    response_format: str
    tool_choice: str
    parallel_tool_calls: bool
    
    