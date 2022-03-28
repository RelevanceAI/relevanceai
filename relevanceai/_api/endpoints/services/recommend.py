"""Recommmend services.
"""
from typing import Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base


class RecommendClient(_Base):
    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def vector(
        self,
        dataset_id: str,
        positive_document_ids: Optional[dict] = None,
        negative_document_ids: Optional[dict] = None,
        vector_fields: Optional[list] = None,
        approximation_depth: int = 0,
        vector_operation: str = "sum",
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: Optional[list] = None,
        filters: Optional[list] = None,
        min_score: float = 0,
        select_fields: Optional[list] = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
    ):
        """
        Vector Search based recommendations are done by extracting the vectors of the documents ids specified performing some vector operations and then searching the dataset with the resultant vector. This allows us to not only do recommendations but personalized and weighted recommendations. \n
        Here are a couple of different scenarios and what the queries would look like for those: \n

        Recommendations Personalized by single liked product:

        >>> positive_document_ids=['A']

        -> Document ID A Vector = Search Query \n

        Recommendations Personalized by multiple liked product:

        >>> positive_document_ids=['A', 'B']

        -> Document ID A Vector + Document ID B Vector = Search Query \n

        Recommendations Personalized by multiple liked product and disliked products:

        >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D']

        -> (Document ID A Vector + Document ID B Vector) - (Document ID C Vector + Document ID C Vector) = Search Query \n

        Recommendations Personalized by multiple liked product and disliked products with weights:

        >>> positive_document_ids={'A':0.5, 'B':1}, negative_document_ids={'C':0.6, 'D':0.4}

        -> (Document ID A Vector * 0.5 + Document ID B Vector * 1) - (Document ID C Vector * 0.6 + Document ID D Vector * 0.4) = Search Query \n

        You can change the operator between vectors with vector_operation:

        e.g. >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D'], vector_operation='multiply'

        -> (Document ID A Vector * Document ID B Vector) - (Document ID C Vector * Document ID D Vector) = Search Query

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        vector_fields: list
            The vector field to search in. It can either be an array of strings (automatically equally weighted) (e.g. ['check_vector_', 'yellow_vector_']) or it is a dictionary mapping field to float where the weighting is explicitly specified (e.g. {'check_vector_': 0.2, 'yellow_vector_': 0.5})
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        vector_operation: string
            Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        """
        positive_document_ids = (
            {} if positive_document_ids is None else positive_document_ids
        )
        negative_document_ids = (
            {} if negative_document_ids is None else negative_document_ids
        )
        vector_fields = [] if vector_fields is None else vector_fields
        facets = [] if facets is None else facets
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        return self.make_http_request(
            f"/services/recommend/vector",
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
                "hundred_scale": hundred_scale,
            },
        )

    def diversity(
        self,
        dataset_id: str,
        cluster_vector_field: str,
        n_clusters: int,
        positive_document_ids: Optional[dict] = None,
        negative_document_ids: Optional[dict] = None,
        vector_fields: Optional[list] = None,
        approximation_depth: int = 0,
        vector_operation: str = "sum",
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: Optional[list] = None,
        filters: Optional[list] = None,
        min_score: float = 0,
        select_fields: Optional[list] = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
        search_history_id: str = None,
        n_init: int = 5,
        n_iter: int = 10,
        return_as_clusters: bool = False,
    ):
        positive_document_ids = (
            {} if positive_document_ids is None else positive_document_ids
        )
        negative_document_ids = (
            {} if negative_document_ids is None else negative_document_ids
        )
        vector_fields = [] if vector_fields is None else vector_fields
        facets = [] if facets is None else facets
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields
        """
        Vector Search based recommendations are done by extracting the vectors of the documents ids specified performing some vector operations and then searching the dataset with the resultant vector. This allows us to not only do recommendations but personalized and weighted recommendations. \n
        Diversity recommendation increases the variety within the recommendations via clustering. Search results are clustered and the top k items in each cluster are selected. The main clustering parameters are cluster_vector_field and n_clusters, the vector field on which to perform clustering and number of clusters respectively. \n
        Here are a couple of different scenarios and what the queries would look like for those: \n

        Recommendations Personalized by single liked product:

        >>> positive_document_ids=['A']

        -> Document ID A Vector = Search Query

        Recommendations Personalized by multiple liked product:

        >>> positive_document_ids=['A', 'B']

        -> Document ID A Vector + Document ID B Vector = Search Query

        Recommendations Personalized by multiple liked product and disliked products:

        >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D']

        -> (Document ID A Vector + Document ID B Vector) - (Document ID C Vector + Document ID C Vector) = Search Query

        Recommendations Personalized by multiple liked product and disliked products with weights:

        >>> positive_document_ids={'A':0.5, 'B':1}, negative_document_ids={'C':0.6, 'D':0.4}

        -> (Document ID A Vector * 0.5 + Document ID B Vector * 1) - (Document ID C Vector * 0.6 + Document ID D Vector * 0.4) = Search Query

        You can change the operator between vectors with vector_operation:

        e.g. >>> positive_document_ids=['A', 'B'], negative_document_ids=['C', 'D'], vector_operation='multiply'

        -> (Document ID A Vector * Document ID B Vector) - (Document ID C Vector * Document ID D Vector) = Search Query

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        cluster_vector_field: str
            The field to cluster on.
        n_clusters: int
            Number of clusters to be specified.
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        vector_fields: list
            The vector field to search in. It can either be an array of strings (automatically equally weighted) (e.g. ['check_vector_', 'yellow_vector_']) or it is a dictionary mapping field to float where the weighting is explicitly specified (e.g. {'check_vector_': 0.2, 'yellow_vector_': 0.5})
        approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        vector_operation: string
            Aggregation for the vectors when using positive and negative document IDs, choose from ['mean', 'sum', 'min', 'max', 'divide', 'mulitple']
        sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        include_vector: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        asc: bool
            Whether to sort results by ascending or descending order
        keep_search_history: bool
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        search_history_id: str
            Search history ID, only used for storing search histories.
        n_init: int
            Number of runs to run with different centroid seeds
        n_iter: int
            Number of iterations in each run
        return_as_clusters: bool
            If True, return as clusters as opposed to results list
        """

        return self.make_http_request(
            f"/services/recommend/diversity",
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
                "hundred_scale": hundred_scale,
                "search_history_id": search_history_id,
                "cluster_vector_field": cluster_vector_field,
                "n_clusters": n_clusters,
                "n_init": n_init,
                "n_iter": n_iter,
                "return_as_clusters": return_as_clusters,
            },
        )
