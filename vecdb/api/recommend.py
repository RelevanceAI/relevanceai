"""Recommmend services.
"""
from ..base import Base


class Recommend(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def vector(
        self,
        dataset_id: str,
        positive_document_ids: dict = {},
        negative_document_ids: dict = {},
        vector_fields=[],
        approximation_depth: int = 0,
        vector_operation: str = "sum",
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: list = [],
        filters: list = [],
        min_score: float = 0,
        select_fields: list = [],
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
    ):
        return self.make_http_request(
            f"services/recommend/recommend/vector",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "positive_document_ids": positive_document_ids,
                "negative_document_ids": negative_document_ids,
                "vector_fields": vector_fields,
                "approximation_depth": approximation_depth,
                "vector_operation": vector_operation,
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
                "keep_search_history": keep_search_history,
            },
        )
