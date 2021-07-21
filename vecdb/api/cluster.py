from ..base import Base
from .centroids import Centroids

class Cluster(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        self.centroids = Centroids(project=project, api_key=api_key, base_url=base_url)

    def aggregate(self, dataset_id: str, vector_field: str, metrics: list,  groupby: list = [], filters: list = [], 
                page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True, output_format: str = "json"):
        return self.make_http_request(
            endpoint="services/cluster/aggregate",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "aggregation_query": {
                    "groupby": groupby,
                    "metrics": metrics
                },
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "flatten": flatten,
                "vector_field": vector_field
            }, output_format = output_format)
        
        
        
        