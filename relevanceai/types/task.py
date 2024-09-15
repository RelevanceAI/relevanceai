from pydantic import BaseModel

task = {
  "id": "task_abc123",
  "object": "task",
  "created_at": 1698107661,
  "metadata": {}
}

class JobInfo(BaseModel):
    studio_id: str
    job_id: str

class Task(BaseModel):
    conversation_id: str
    job_info: JobInfo
    agent_id: str
    state: str