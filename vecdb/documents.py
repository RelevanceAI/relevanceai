from .base import Base

class Documents(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        
    def list(self, dataset_id: str):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/list", 
            method="GET")
        
    def get(self, dataset_id: str, id: str, select_fields: list=[],
        cursor: str=None, page_size: int=20, sort: list=[],
        include_vector: bool=True):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get",
            parameters={
                "id": id,
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector
            }
        )
