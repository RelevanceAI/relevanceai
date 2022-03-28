"""Tagger services
"""
from typing import Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base


class TaggerClient(_Base):
    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def tag(
        self,
        data: str,
        tag_dataset_id: str,
        encoder: str,
        tag_field: str = None,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[list] = None,
        min_score: float = 0,
        include_search_relevance: bool = False,
        search_relevance_cutoff_aggressiveness: int = 1,
        asc: bool = False,
        include_score: bool = False,
    ):
        """
        Tag documents or vectors

        Parameters
        ----------
        data : string
            Image Url or text or any data suited for the encoder
        tag_dataset_id : string
            Name of the dataset you want to tag
        encoder: string
            Which encoder to use.
        tag_field: string
            The field used to tag in a dataset. If None, automatically uses the one stated in the encoder.
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        include_search_relevance: bool
            Whether to calculate a search_relevance cutoff score to flag relevant and less relevant results
        search_relevance_cutoff_aggressiveness: int
            How aggressive the search_relevance cutoff score is (higher value the less results will be relevant)
        asc: bool
            Whether to sort results by ascending or descending order
        include_score: bool
            Whether to include score
        """
        filters = [] if filters is None else filters

        return self.make_http_request(
            f"/services/tagger/tag",
            method="POST",
            parameters={
                "data": data,
                "tag_dataset_id": tag_dataset_id,
                "encoder": encoder,
                "tag_field": tag_field,
                "approximation_depth": approximation_depth,
                "sum_fields": sum_fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "filters": filters,
                "min_score": min_score,
                "include_search_relevance": include_search_relevance,
                "search_relevance_cutoff_aggressiveness": search_relevance_cutoff_aggressiveness,
                "asc": asc,
                "include_score": include_score,
            },
        )

    def diversity(
        self,
        data: str,
        tag_dataset_id: str,
        encoder: str,
        cluster_vector_field: str,
        n_clusters: int,
        tag_field: str = None,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[list] = None,
        min_score: float = 0,
        include_search_relevance: bool = False,
        search_relevance_cutoff_aggressiveness: int = 1,
        asc: bool = False,
        include_score: bool = False,
        n_init: int = 5,
        n_iter: int = 10,
    ):
        """
        Tagging and then clustering the tags and returning one from each cluster (starting from the closest tag)

        Parameters
        ----------
        data : string
            Image Url or text or any data suited for the encoder
        tag_dataset_id : string
            Name of the dataset you want to tag
        encoder: string
            Which encoder to use.
        cluster_vector_field: str
            The field to cluster on.
        n_clusters: int
            Number of clusters to be specified.
        tag_field: string
            The field used to tag in a dataset. If None, automatically uses the one stated in the encoder.
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        include_search_relevance: bool
            Whether to calculate a search_relevance cutoff score to flag relevant and less relevant results
        search_relevance_cutoff_aggressiveness: int
            How aggressive the search_relevance cutoff score is (higher value the less results will be relevant)
        asc: bool
            Whether to sort results by ascending or descending order
        include_score: bool
            Whether to include score
        n_init: int
            Number of runs to run with different centroid seeds
        n_iter: int
            Number of iterations in each run
        """
        filters = [] if filters is None else filters

        return self.make_http_request(
            f"/services/tagger/diversity",
            method="POST",
            parameters={
                "data": data,
                "tag_dataset_id": tag_dataset_id,
                "encoder": encoder,
                "tag_field": tag_field,
                "approximation_depth": approximation_depth,
                "sum_fields": sum_fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "filters": filters,
                "min_score": min_score,
                "include_search_relevance": include_search_relevance,
                "search_relevance_cutoff_aggressiveness": search_relevance_cutoff_aggressiveness,
                "asc": asc,
                "include_score": include_score,
                "cluster_vector_field": cluster_vector_field,
                "n_clusters": n_clusters,
                "n_init": n_init,
                "n_iter": n_iter,
            },
        )
