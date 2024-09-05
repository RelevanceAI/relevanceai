from pydantic import BaseModel
from typing import List, Optional

run_step = {
  "id": "step_abc123",
  "object": "task.run.step",
  "created_at": 1699063291,
  "run_id": "run_abc123",
  "assistant_id": "asst_abc123",
  "task_id": "task_abc123",
  "type": "message_creation",
  "status": "completed",
  "cancelled_at": None,
  "completed_at": 1699063291,
  "expired_at": None,
  "failed_at": None,
  "last_error": None,
  "step_details": {
    "type": "message_creation",
    "message_creation": {
      "message_id": "msg_abc123"
    }
  },
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
}

class Step(BaseModel):
    id: str
    object: str
    created_at: int
    run_id: str
    assistant_id: str
    task_id: str
    type: str
    status: str
    cancelled_at: Optional[int]
    completed_at: Optional[int]
    expired_at: Optional[int]
    failed_at: Optional[int]
    last_error: Optional[str]
    step_details: dict
    usage: dict
    
