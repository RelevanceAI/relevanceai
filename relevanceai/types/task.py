from pydantic import BaseModel

class JobInfo(BaseModel):
    studio_id: str
    job_id: str

class Task(BaseModel):
    conversation_id: str
    job_info: JobInfo
    agent_id: str
    state: str

    def __repr__(self):
        return f"<Task - {self.conversation_id}>"
    

conversation = {'knowledge_set': '04c82c75-c9b6-430f-befe-cae2c89b432c', 'metadata': {'_id': '587fc94b-50fa-4653-947f-a3c5d1a5e787_-_04c82c75-c9b6-430f-befe-cae2c89b432c', 'conversation': {...}, 'field_metadata': {...}, 'hidden': True, 'insert_date': '2024-09-17T07:42:59.822Z', 'insert_datetime': '2024-09-17T07:42:59.822Z', 'knowledge_set': '04c82c75-c9b6-430f-befe-cae2c89b432c', 'project': '587fc94b-50fa-4653-947f-a3c5d1a5e787', 'type': 'conversation', 'update_date': '2024-09-17T07:42:59.821Z', 'update_datetime': '2024-09-17T07:42:59.821Z'}}

class Conversation(BaseModel):
    pass