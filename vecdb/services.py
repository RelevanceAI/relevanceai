"""Services class
"""
from .base import Base
from .encoders import Encoders
from .cluster import Cluster

class Services(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
        self.encoders = Encoders(project=project, api_key=api_key, 
            base_url=base_url)
        self.cluster = Cluster(project=project, api_key=api_key, base_url=base_url)
    
    def search(self, dataset_id: str, multivector_query: list, positive_document_ids: dict={},
        negative_document_ids: dict={}, vector_operation="sum", approximation_depth=0,
        sum_fields=True, page_size=20, page=1, similarity_metric="cosine", facets=[], filters=[],
        min_score=0, select_fields=[], include_vector=False, include_count=True, asc=False, keep_search_history=False):
        return self.make_http_request("services/search/vector", method="POST", parameters=
            {"dataset_id": dataset_id,
            "multivector_query": multivector_query,
            "positive_document_ids": positive_document_ids,
            "negative_document_ids": negative_document_ids,
            "vector_operation": vector_operation,
            "approximation_depth": approximation_depth,
            "sum_fields": sum_fields,
            "page_size": page_size,
            "page": page,
            "similarity_metric": similarity_metric,
            "facets": facets,
            "filters": filters,
            "min_score": min_score,
            "select_fields": select_fields,
            "include_vector": include_vector,
            "include_count": include_count,
            "asc": asc,
            "keep_search_history": keep_search_history
            })
        
    def hybrid(self, dataset_id: str, vector_query: list, query: str, fields:list, page_size: int):
        return self.make_http_request("services/search/hybrid", method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": vector_query,
                "text": query,
                "fields": fields,
                "page_size": page_size
            })
