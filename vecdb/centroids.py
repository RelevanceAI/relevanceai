from .base import Base

class Centroids(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def list(self, dataset_id: str, vector_field: str, alias: str="default",
        page_size: int=5, cursor: str=None, include_vector: bool=False):
        return self.make_http_request(
            "services/cluster/centroids/list",
            method="GET",
            parameters={
                "dataset_id": dataset_id,
                "vector_field": vector_field,
                "alias": alias,
                "page_size": page_size,
                "cursor": cursor,
                "include_vector": include_vector
            }
        )
    
    def get(self, dataset_id: str, cluster_ids: list, vector_field: str,
        alias: str="default", page_size: int=5, cursor: str=None):
        return self.make_http_request(
            "services/cluster/centroids/get",
            method="GET",
            parameters={
                "dataset_id": dataset_id,
                "cluster_ids": cluster_ids,
                "vector_field": vector_field,
                "alias": alias,
                "page_size": page_size,
                "cursor": cursor
            }
        )
