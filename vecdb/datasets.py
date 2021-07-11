"""All Dataset related functions
"""
from .base import Base
from .tasks import Tasks
from .documents import Documents

class Datasets(Base):
    """All dataset-related functions
    """
    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
        self.tasks = Tasks(project=project, api_key=api_key, 
            base_url=base_url)
        self.documents = Documents(project=project, api_key=api_key,
            base_url=base_url)

    def get_where(self, dataset_id: str, filters: list=[], cursor: str=None, 
        page_size: int=20, sort: list=[], select_fields: list=[], 
        include_vector: bool=True):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get_where", 
            method="POST", 
            parameters={
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
                "filters": filters})
    
    def schema(self, dataset_id: str):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/schema", method="GET")

    def list(self):
        return self.make_http_request(endpoint=f"datasets/list", method="GET")
    
    def facets(self, dataset_id, fields: list, date_interval: str="monthly", 
        page_size: int=5, page: int=1, asc: bool=False):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/facets",
            method="POST",
            parameters={
                "fields": fields,
                "date_interval": date_interval,
                "page_size": page_size,
                "page": page,
                "asc": asc
            })
    
    def delete(self, dataset_id):
        return self.make_http_request(endpoint=f"datasets/delete",
            method="POST",
            parameters={
                "dataset_id": dataset_id  
            })
