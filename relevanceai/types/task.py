from pydantic import BaseModel

class JobInfo(BaseModel):
    studio_id: str
    job_id: str

class Task(BaseModel):
    conversation_id: str
    job_info: JobInfo
    agent_id: str
    state: str