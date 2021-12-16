from relevanceai.base import _Base
from typing import List


class SearchClient(_Base):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)
        self._init_experiment_helper()

    def vector(
        self,
        dataset_id: str,
        multivector_query: List,
        positive_document_ids: dict = {},
        negative_document_ids: dict = {},
        vector_operation="sum",
        approximation_depth=0,
        sum_fields=True,
        page_size=20,
        page=1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
        query: str = None,
    ):
        """
        Allows you to leverage vector similarity search to create a semantic search engine. Powerful features of VecDB vector search:

        1. Multivector search that allows you to search with multiple vectors and give each vector a different weight. e.g. Search with a product image vector and text description vector to find the most similar products by what it looks like and what its described to do. You can also give weightings of each vector field towards the search, e.g. image_vector_ weights 100%, whilst description_vector_ 50% \n
            An example of a simple multivector query:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": ["name_vector_"], "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": ["image_vector_"], "alias":"image"},
            >>> ]

            An example of a weighted multivector query:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": {"name_vector_":0.6}, "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": {"image_vector_"0.4}, "alias":"image"},
            >>> ]

            An example of a weighted multivector query with multiple fields for each vector:

            >>> [
            >>>     {"vector": [0.12, 0.23, 0.34], "fields": {"name_vector_":0.6, "description_vector_":0.3}, "alias":"text"},
            >>>     {"vector": [0.45, 0.56, 0.67], "fields": {"image_vector_"0.4}, "alias":"image"},
            >>> ]

        2. Utilise faceted search with vector search. For information on how to apply facets/filters check out datasets.documents.get_where \n
        3. Sum Fields option to adjust whether you want multiple vectors to be combined in the scoring or compared in the scoring. e.g. image_vector_ + text_vector_ or image_vector_ vs text_vector_. \n
            When sum_fields=True:

            - Multi-vector search allows you to obtain search scores by taking the sum of these scores.
            - TextSearchScore + ImageSearchScore = SearchScore
            - We then rank by the new SearchScore, so for searching 1000 documents there will be 1000 search scores and results

            When sum_fields=False:

            - Multi vector search but not summing the score, instead including it in the comparison!
            - TextSearchScore = SearchScore1
            - ImagSearchScore = SearchScore2
            - We then rank by the 2 new SearchScore, so for searching 1000 documents there should be 2000 search scores and results.

        4. Personalization with positive and negative document ids.

            - For more information about the positive and negative document ids to personalize check out services.recommend.vector

        For more even more advanced configuration and customisation of vector search, reach out to us at dev@relevance.ai and learn about our new advanced_vector_search.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
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
        search_history_id: string
            Search history ID, only used for storing search histories.
        query: string
            What to store as the query name in the dashboard
        """
        return self.make_http_request(
            "/services/search/vector",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
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
                "keep_search_history": keep_search_history,
                "hundred_scale": hundred_scale,
                "search_history_id": search_history_id,
                "query": query,
            },
        )

    def hybrid(
        self,
        dataset_id: str,
        multivector_query: List,
        text: str,
        fields: list,
        edit_distance: int = -1,
        ignore_spaces: bool = True,
        traditional_weight: float = 0.075,
        page_size: int = 20,
        page=1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
    ):
        """
        Combine the best of both traditional keyword faceted search with semantic vector search to create the best search possible. \n

        For information on how to use vector search check out services.search.vector. \n

        For information on how to use traditional keyword faceted search check out services.search.traditional.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        text : string
            Text Search Query (not encoded as vector)
        fields : list
            Text fields to search against
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
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
        min_score: float
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
        search_history_id: string
            Search history ID, only used for storing search histories.
        edit_distance: int
            This refers to the amount of letters it takes to reach from 1 string to another string. e.g. band vs bant is a 1 word edit distance. Use -1 if you would like this to be automated.
        ignore_spaces: bool
            Whether to consider cases when there is a space in the word. E.g. Go Pro vs GoPro.
        traditional_weight: int
            Multiplier of traditional search score. A value of 0.025~0.075 is the ideal range
        """
        return self.make_http_request(
            "/services/search/hybrid",
            method="POST",
            parameters={
                "dataset_id": "ecommerce-experiments",
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "text": text,
                "fields": fields,
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
                "edit_distance": edit_distance,
                "ignore_spaces": ignore_spaces,
                "traditional_weight": traditional_weight,
                "query": text,
            },
        )

    def semantic(
        self,
        dataset_id: str,
        multivector_query: list,
        fields: list,
        text: str,
        page_size: int = 20,
        page=1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
    ):
        """
        A more automated hybrid search with a few extra things that automatically adjusts some of the key parameters for more automated and good out of the box results. \n

        For information on how to configure semantic search check out services.search.hybrid.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        text : string
            Text Search Query (not encoded as vector)
        fields : list
            Text fields to search against
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
        return self.make_http_request(
            "/services/search/semantic",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "text": text,
                "fields": fields,
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
                "query": text,
            },
        )

    def diversity(
        self,
        dataset_id: str,
        cluster_vector_field: str,
        n_clusters: int,
        multivector_query: list,
        positive_document_ids: dict = {},
        negative_document_ids: dict = {},
        vector_operation="sum",
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric="cosine",
        facets=[],
        filters=[],
        min_score=0,
        select_fields=[],
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale: bool = False,
        search_history_id: str = None,
        n_init: int = 5,
        n_iter: int = 10,
        return_as_clusters: bool = False,
        query: str = None,
    ):
        """
        This will first perform an advanced search and then cluster the top X (page_size) search results. Results are returned as such: Once you have the clusters:

        >>> Cluster 0: [A, B, C]
        >>> Cluster 1: [D, E]
        >>> Cluster 2: [F, G]
        >>> Cluster 3: [H, I]

        (Note, each cluster is ordered by highest to lowest search score.) \n

        This intermediately returns:

        >>> results_batch_1: [A, H, F, D] (ordered by highest search score)
        >>> results_batch_2: [G, E, B, I] (ordered by highest search score)
        >>> results_batch_3: [C]

        This then returns the final results:

        >>> results: [A, H, F, D, G, E, B, I, C]

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        cluster_vector_field: str
            The field to cluster on.
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        positive_document_ids : dict
            Positive document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
        negative_document_ids: dict
            Negative document IDs to personalize the results with, this will retrive the vectors from the document IDs and consider it in the operation.
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
        n_clusters: int
            Number of clusters to be specified.
        n_init: int
            Number of runs to run with different centroid seeds
        n_iter: int
            Number of iterations in each run
        return_as_clusters: bool
            If True, return as clusters as opposed to results list
        query: string
            What to store as the query name in the dashboard
        """
        return self.make_http_request(
            "/services/search/diversity",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
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
                "keep_search_history": keep_search_history,
                "hundred_scale": hundred_scale,
                "search_history_id": search_history_id,
                "cluster_vector_field": cluster_vector_field,
                "n_clusters": n_clusters,
                "n_init": n_init,
                "n_iter": n_iter,
                "return_as_clusters": return_as_clusters,
                "query": query,
            },
        )

    def traditional(
        self,
        dataset_id: str,
        text: str,
        fields: list = [],
        edit_distance: int = -1,
        ignore_spaces: bool = True,
        page_size: int = 29,
        page: int = 1,
        select_fields: list = [],
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        search_history_id: str = None,
    ):
        """
        Traditional Faceted Keyword Search with edit distance/fuzzy matching. \n

        For information on how to apply facets/filters check out datasets.documents.get_where. \n

        For information on how to construct the facets section for your search bar check out datasets.facets.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        text : string
            Text Search Query (not encoded as vector)
        fields : list
            Text fields to search against
        edit_distance: int
            This refers to the amount of letters it takes to reach from 1 string to another string. e.g. band vs bant is a 1 word edit distance. Use -1 if you would like this to be automated.
        ignore_spaces: bool
            Whether to consider cases when there is a space in the word. E.g. Go Pro vs GoPro.
        page_size: int
            Size of each page of results
        page: int
            Page of the results
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
        search_history_id: string
            Search history ID, only used for storing search histories.
        """
        return self.make_http_request(
            "/services/search/traditional",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "text": text,
                "fields": fields,
                "edit_distance": edit_distance,
                "ignore_spaces": ignore_spaces,
                "page_size": page_size,
                "page": page,
                "select_fields": select_fields,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history,
                "search_history_id": search_history_id,
                "query": "text",
            },
        )

    def chunk(
        self,
        dataset_id,
        multivector_query,
        chunk_field,
        chunk_scoring="max",
        chunk_page_size: int = 3,
        chunk_page: int = 1,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: list = [],
        filters: list = [],
        min_score: int = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
        query: str = None,
    ):

        """
        Chunks are data that has been divided into different units. e.g. A paragraph is made of many sentence chunks, a sentence is made of many word chunks, an image frame in a video. By searching through chunks you can pinpoint more specifically where a match is occuring. When creating a chunk in your document use the suffix "chunk" and "chunkvector". An example of a document with chunks:

        >>> {
        >>>     "_id" : "123",
        >>>     "title" : "Lorem Ipsum Article",
        >>>     "description" : "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.",
        >>>     "description_vector_" : [1.1, 1.2, 1.3],
        >>>     "description_sentence_chunk_" : [
        >>>         {"sentence_id" : 0, "sentence_chunkvector_" : [0.1, 0.2, 0.3], "sentence" : "Lorem Ipsum is simply dummy text of the printing and typesetting industry."},
        >>>         {"sentence_id" : 1, "sentence_chunkvector_" : [0.4, 0.5, 0.6], "sentence" : "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book."},
        >>>         {"sentence_id" : 2, "sentence_chunkvector_" : [0.7, 0.8, 0.9], "sentence" : "It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged."},
        >>>     ]
        >>> }

        For combining chunk search with other search check out services.search.advanced_chunk.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        chunk_field : string
            Field where the array of chunked documents are.
        chunk_scoring: string
            Scoring method for determining for ranking between document chunks.
        chunk_page_size: int
            Size of each page of chunk results
        chunk_page: int
            Page of the chunk results
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
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
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
        query: string
            What to store as the query name in the dashboard
        """

        return self.make_http_request(
            "/services/search/chunk",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "chunk_field": chunk_field,
                "chunk_scoring": chunk_scoring,
                "chunk_page_size": chunk_page_size,
                "chunk_page": chunk_page,
                "approximation_depth": approximation_depth,
                "sum_fields": sum_fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "facets": facets,
                "filters": filters,
                "min_score": min_score,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history,
                "hundred_scale": hundred_scale,
                "query": query,
            },
        )

    def multistep_chunk(
        self,
        dataset_id,
        multivector_query,
        first_step_multivector_query,
        chunk_field,
        chunk_scoring="max",
        chunk_page_size: int = 3,
        chunk_page: int = 1,
        approximation_depth: int = 0,
        sum_fields: bool = True,
        page_size: int = 20,
        page: int = 1,
        similarity_metric: str = "cosine",
        facets: list = [],
        filters: list = [],
        min_score: int = None,
        include_vector: bool = False,
        include_count: bool = True,
        asc: bool = False,
        keep_search_history: bool = False,
        hundred_scale: bool = False,
        first_step_page: int = 1,
        first_step_page_size: int = 20,
        query: str = None,
    ):

        """
        Multistep chunk search involves a vector search followed by chunk search, used to accelerate chunk searches or to identify context before delving into relevant chunks. e.g. Search against the paragraph vector first then sentence chunkvector after. \n

        For more information about chunk search check out services.search.chunk. \n

        For more information about vector search check out services.search.vector

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        multivector_query : list
            Query for advance search that allows for multiple vector and field querying.
        chunk_field : string
            Field where the array of chunked documents are.
        chunk_scoring: string
            Scoring method for determining for ranking between document chunks.
        chunk_page_size: int
            Size of each page of chunk results
        chunk_page: int
            Page of the chunk results
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
        facets: list
            Fields to include in the facets, if [] then all
        filters: list
            Query for filtering the search results
        min_score: int
            Minimum score for similarity metric
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
        first_step_multivector_query: list
            Query for advance search that allows for multiple vector and field querying.
        first_step_page: int
            Page of the results
        first_step_page_size: int
            Size of each page of results
        query: string
            What to store as the query name in the dashboard
        """
        return self.make_http_request(
            "/services/search/multistep_chunk",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "multivector_query": multivector_query,
                "chunk_field": chunk_field,
                "chunk_scoring": chunk_scoring,
                "chunk_page_size": chunk_page_size,
                "chunk_page": chunk_page,
                "approximation_depth": approximation_depth,
                "sum_fields": sum_fields,
                "page_size": page_size,
                "page": page,
                "similarity_metric": similarity_metric,
                "facets": facets,
                "filters": filters,
                "min_score": min_score,
                "include_vector": include_vector,
                "include_count": include_count,
                "asc": asc,
                "keep_search_history": keep_search_history,
                "hundred_scale": hundred_scale,
                "first_step_multivector_query": first_step_multivector_query,
                "first_step_page": first_step_page,
                "first_step_page_size": first_step_page_size,
                "query": query,
            },
        )

    def advanced_chunk(
        self,
        dataset_ids,
        chunk_search_query: List,
        min_score: int = None,
        page_size: int = 20,
        include_vector: bool = False,
        select_fields: list = [],
        query: str = None,
    ):
        """
        A more advanced chunk search to be able to combine vector search and chunk search in many different ways. \n
        Example 1 (Hybrid chunk search):
        >>> chunk_query = {
        >>>     "chunk" : "some.test",
        >>>     "queries" : [
        >>>         {"vector" : vec1, "fields": {"some.test.some_chunkvector_":1},
        >>>         "traditional_query" : {"text":"python", "fields" : ["some.test.test_words"], "traditional_weight": 0.3},
        >>>         "metric" : "cosine"},
        >>>         {"vector" : vec, "fields": ["some.test.tt.some_other_chunkvector_"],
        >>>         "traditional_query" : {"text":"jumble", "fields" : ["some.test.test_words"], "traditional_weight": 0.3},
        >>>         "metric" : "cosine"},
        >>>     ]
        >>> }

        Example 2 (combines normal vector search with chunk search):
        >>> chunk_query = {
        >>>     "queries" : [
        >>>         {
        >>>             "queries": [
        >>>                 {
        >>>                     "vector": vec1,
        >>>                     "fields": {
        >>>                         "some.test.some_chunkvector_": 0.9
        >>>                     },
        >>>                     "traditional_query": {
        >>>                         "text": "python",
        >>>                         "fields": [
        >>>                             "some.test.test_words"
        >>>                         ],
        >>>                         "traditional_weight": 0.3
        >>>                     },
        >>>                     "metric": "cosine"
        >>>                 }
        >>>             ],
        >>>             "chunk": "some.test",
        >>>         },
        >>>         {
        >>>             "vector" : vec,
        >>>             "fields": {
        >>>                 ".some_vector_" : 0.1},
        >>>                 "metric" : "cosine"
        >>>                 },
        >>>         ]
        >>>     }

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        chunk_search_query : list
            Advanced chunk query
        min_score: int
            Minimum score for similarity metric
        page_size: int
            Size of each page of results
        include_vector: bool
            Include vectors in the search results
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        query: string
            What to store as the query name in the dashboard
        """
        return self.make_http_request(
            "/services/search/advanced_chunk",
            method="POST",
            parameters={
                "dataset_ids": dataset_ids,
                "chunk_search_query": chunk_search_query,
                "page_size": page_size,
                "min_score": min_score,
                "include_vector": include_vector,
                "select_fields": select_fields,
                "query": query,
            },
        )

    def advanced_multistep_chunk(
        self,
        dataset_ids: list,
        first_step_query: list,
        first_step_text: str,
        first_step_fields: list,
        chunk_search_query: list,
        first_step_edit_distance: int = -1,
        first_step_ignore_space: bool = True,
        first_step_traditional_weight: float = 0.075,
        first_step_approximation_depth: int = 0,
        first_step_sum_fields: bool = True,
        first_step_filters: list = [],
        first_step_page_size: int = 50,
        include_count: bool = True,
        min_score: int = 0,
        page_size: int = 20,
        include_vector: bool = False,
        select_fields: list = [],
        query: str = None,
    ):
        """
        Performs a vector hybrid search and then an advanced chunk search. Chunk Search allows one to search through chunks inside a document. The major difference between chunk search and normal search in Vector AI is that it relies on the chunkvector field. Chunk Vector Search. Search with a multiple chunkvectors for the most similar documents. Chunk search also supports filtering to only search through filtered results and facets to get the overview of products available when a minimum score is set. \n

        Example 1 (Hybrid chunk search):

        >>> chunk_query = {
        >>>     "chunk" : "some.test",
        >>>     "queries" : [
        >>>         {"vector" : vec1, "fields": {"some.test.some_chunkvector_":1},
        >>>         "traditional_query" : {"text":"python", "fields" : ["some.test.test_words"], "traditional_weight": 0.3},
        >>>         "metric" : "cosine"},
        >>>         {"vector" : vec, "fields": ["some.test.tt.some_other_chunkvector_"],
        >>>         "traditional_query" : {"text":"jumble", "fields" : ["some.test.test_words"], "traditional_weight": 0.3},
        >>>         "metric" : "cosine"},
        >>>     ]
        >>> }

        Example 2 (combines normal vector search with chunk search):
        >>> chunk_query = {
        >>>     "queries" : [
        >>>         {
        >>>             "queries": [
        >>>                 {
        >>>                     "vector": vec1,
        >>>                     "fields": {
        >>>                         "some.test.some_chunkvector_": 0.9
        >>>                     },
        >>>                     "traditional_query": {
        >>>                         "text": "python",
        >>>                         "fields": [
        >>>                             "some.test.test_words"
        >>>                         ],
        >>>                         "traditional_weight": 0.3
        >>>                     },
        >>>                     "metric": "cosine"
        >>>                 }
        >>>             ],
        >>>             "chunk": "some.test",
        >>>         },
        >>>         {
        >>>             "vector" : vec,
        >>>             "fields": {
        >>>                 ".some_vector_" : 0.1},
        >>>                 "metric" : "cosine"
        >>>                 },
        >>>         ]
        >>>     }

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        first_step_query : list
            First step query
        first_step_text : string
            Text search query (not encoded as vector)
        first_step_fields: list
            Text fields to search against
        chunk_search_query: list
            Advanced chunk query
        first_step_edit_distance: int
            This refers to the amount of letters it takes to reach from 1 string to another string. e.g. band vs bant is a 1 word edit distance. Use -1 if you would like this to be automated.
        first_step_ignore_spaces: bool
            Whether to consider cases when there is a space in the word. E.g. Go Pro vs GoPro.
        first_step_traditional_weight: int
            Multiplier of traditional search score. A value of 0.025~0.075 is the ideal range
        first_step_approximation_depth: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate.
        first_step_sum_fields : bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        first_step_filters: list
            Query for filtering the search results
        first_step_page_size: int
            In the first search, you are more interested in the contents
        include_count: bool
            Include the total count of results in the search results
        min_score: int
            Minimum score for similarity metric
        page_size: int
            Size of each page of results
        include_vector: bool
            Include vectors in the search results
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        query: string
            What to store as the query name in the dashboard
        """
        return self.make_http_request(
            "/services/search/advanced_multistep_chunk",
            method="POST",
            parameters={
                "dataset_ids": dataset_ids,
                "first_step_query": first_step_query,
                "first_step_text": first_step_text,
                "first_step_fields": first_step_fields,
                "chunk_search_query": chunk_search_query,
                "first_step_edit_distance": first_step_edit_distance,
                "first_step_ignore_space": first_step_ignore_space,
                "first_step_traditional_weight": first_step_traditional_weight,
                "first_step_approximation_depth": first_step_approximation_depth,
                "first_step_sum_fields": first_step_sum_fields,
                "first_step_filters": first_step_filters,
                "first_step_page_size": first_step_page_size,
                "include_count": include_count,
                "min_score": min_score,
                "page_size": page_size,
                "include_vector": include_vector,
                "select_fields": select_fields,
                "query": query,
            },
        )

    def _init_experiment_helper(
        self, categories=["chunk", "vector", "diversity", "traditional"]
    ):
        self.categories = categories
        self.traditional_search_doc = (
            "https://docs.relevance.ai/docs/better-text-search-with-hybrid"
        )
        self.vector_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both"
        self.diversity_search_doc = "https://docs.relevance.ai/docs/better-text-search-diversified-search-results"
        self.hybrid_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both-1"
        self.semantic_search_doc = "https://docs.relevance.ai/docs/pure-word-matching-pure-vector-search-or-combination-of-both-2"
        self.chunk_search_doc = (
            "https://docs.relevance.ai/docs/better-text-search-chunk-search"
        )
        self.multistep_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data"
        self.advanced_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data-1"
        self.advanced_multistep_chunk_doc = "https://docs.relevance.ai/docs/fine-grained-search-search-on-chunks-of-text-data-2"

        self.initiative_messages = "What else to experiment with :)\n"
        self.category_initiative_messages = {
            "chunk": "if you are searching on large pieces of text, you could chunk your data and try\n",
            "vector": "if you are looking for strong conceptual relations and not just word matching, you could try\n",
            "diversity": "if you are looking for strong conceptual relations as well as diverse results, you could try\n",
            "traditional": "if you are looking for specific text such as id, names, etc., you could try\n",
        }

    def make_suggestion(self):
        if hasattr(self, "_last_used_endpoint"):
            self.last_search = self._last_used_endpoint.split("/")[-1]
        else:
            self.last_search = None
        suggestion = self.initiative_messages
        if "traditional" in self.categories and self.last_search != "traditional":
            suggestion += self.category_initiative_messages["traditional"]
            suggestion += f"   * traditional search ({self.traditional_search_doc})\n"

        if "vector" in self.categories:
            suggestion += self.category_initiative_messages["vector"]
            if self.last_search != "vector":
                suggestion += f"   * vector search ({self.vector_search_doc})\n"
            if self.last_search != "hybrid":
                suggestion += f"   * hybrid search ({self.hybrid_search_doc})\n"
            if self.last_search != "semantic":
                suggestion += f"   * semantic search ({self.semantic_search_doc})\n"

        if "diversity" in self.categories and self.last_search != "diversity":
            suggestion += self.category_initiative_messages["diversity"]
            suggestion += f"   * diversity search ({self.diversity_search_doc})\n"

        if "chunk" in self.categories:
            suggestion += self.category_initiative_messages["chunk"]
            if self.last_search != "chunk":
                suggestion += f"   * chunk search ({self.chunk_search_doc})\n"
            if self.last_search != "multistep_chunk":
                suggestion += (
                    f"   * multistep_chunk search ({self.multistep_chunk_doc})\n"
                )
            if self.last_search != "advanced_chunk":
                suggestion += (
                    f"   * advanced_chunk search ({self.advanced_chunk_doc})\n"
                )
            if self.last_search != "advanced_multistep_chunk":
                suggestion += f"   * advanced_multistep_chunk search ({self.advanced_multistep_chunk_doc})\n"

        return {"search": suggestion}
