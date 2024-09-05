from typing import List, Union, Optional
from typing_extensions import Literal

message = {
  "id": "msg_abc123", 
  "object": "task.message",
  "created_at": 1698983503,
  "task_id": "task_abc123",
  "role": "agent", # or user
  "content": [
    {
      "type": "text",
      "text": {
        "value": "Hi! How can I help you today?",
        "annotations": []
      }
    }
  ],
  "agent_id": "agent_abc123",
  "run_id": "run_abc123",
  "attachments": [],
  "metadata": {}
}

class Message:
    id: str
    object: str
    created_at: int
    task_id: str
    role: str
    content: List
    agent_id: str
    run_id: str
    attachments: list
    metadata: dict
    
class MessageDeleted:
    id: str
    deleted: bool 
    object: Literal["task.message.deleted"]