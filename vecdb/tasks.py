"""Tasks Module
"""
from .base import Base

class Tasks(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def create(self, dataset_id, task_name, task_parameters):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": task_name,
                **task_parameters
            }
        )
    
    def status(self, dataset_id: str, task_id: str):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/{task_id}/status",
            method="GET")
