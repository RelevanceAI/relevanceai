"""Search In A Dataset
"""
from typing import List, Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.decorators.version import deprecated_error
from relevanceai._api import APIClient


class SearchOps(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
    ):
        self.dataset_id = dataset_id

        super().__init__(credentials)

    @track
    @deprecated_error("Please use the new `dataset.search` method.")
    def vector_search(
        self,
        multivector_query: List,
        positive_document_ids: Optional[dict] = None,
        negative_document_ids: Optional[dict] = None,
        vector_operation="sum",
        approximation_depth=0,
        sum_fields=True,
        page_size=20,
        page=1,
        similarity_metric="cosine",
        facets: Optional[list] = None,
        filters: Optional[list] = None,
        min_score=0,
        select_fields: Optional[list] = None,
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
        query: str = None,
    ):
        """
        Allows you to leverage vector similarity search to create a semantic search engine. Powerful features of Relevance vector search:

        1. Multivector search that allows you to search with multiple vectors and give each vector a different weight.
        e.g. Search with a product image vector and text description vector to find the most similar products by what it looks like and what its described to do.
        You can also give weightings of each vector field towards the search, e.g. image_vector_ weights 100%, whilst description_vector_ 50% \n
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
            Whether to store the history into Relevance. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        search_history_id: string
            Search history ID, only used for storing search histories.
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """

        positive_document_ids = (
            {} if positive_document_ids is None else positive_document_ids
        )
        negative_document_ids = (
            {} if negative_document_ids is None else negative_document_ids
        )
        facets = [] if facets is None else facets
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        return self.services.search.vector(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            positive_document_ids=positive_document_ids,
            negative_document_ids=negative_document_ids,
            vector_operation=vector_operation,
            approximation_depth=approximation_depth,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            select_fields=select_fields,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            search_history_id=search_history_id,
            query=query,
        )

    @track
    @deprecated_error("Please use the new `dataset.search` method.")
    def hybrid_search(
        self,
        multivector_query: List,
        text: str,
        fields: list,
        edit_distance: int = -1,
        ignore_spaces: bool = True,
        traditional_weight: float = 0.075,
        page_size: int = 20,
        page=1,
        similarity_metric="cosine",
        facets: Optional[list] = None,
        filters: Optional[list] = None,
        min_score=0,
        select_fields: Optional[list] = None,
        include_vector=False,
        include_count=True,
        asc=False,
        keep_search_history=False,
        hundred_scale=False,
        search_history_id=None,
        sum_fields: bool = False,
    ):
        """
        Combine the best of both traditional keyword faceted search with semantic vector search to create the best search possible. \n

        For information on how to use vector search check out services.search.vector. \n

        For information on how to use traditional keyword faceted search check out services.search.traditional.

        Parameters
        ----------
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
            Whether to store the history into Relevance. This will increase the storage costs over time.
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


        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            MULTIVECTOR_QUERY = [{"vector": [0, 1, 2], "fields": ["sample_vector_"]}]
            results = df.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """
        facets = [] if facets is None else facets
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        return self.services.search.hybrid(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            text=text,
            fields=fields,
            edit_distance=edit_distance,
            ignore_spaces=ignore_spaces,
            traditional_weight=traditional_weight,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            select_fields=select_fields,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            search_history_id=search_history_id,
            sum_fields=sum_fields,
        )

    @track
    @deprecated_error("Please use the new `dataset.search` method.")
    def chunk_search(
        self,
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
        facets: Optional[list] = None,
        filters: Optional[list] = None,
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
            Whether to store the history into Relevance. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.chunk_search(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY
            )

        """
        raise DeprecationWarning()  # Should error as above based on decorator

    @track
    @deprecated_error("Please use the new `dataset.search` method.")
    def multistep_chunk_search(
        self,
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
        facets: Optional[list] = None,
        filters: Optional[list] = None,
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

        For more information about chunk search check out datasets.search.chunk. \n

        For more information about vector search check out services.search.vector

        Parameters
        ----------

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
            Whether to store the history into Relevance. This will increase the storage costs over time.
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

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            results = df.search.multistep_chunk(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY,
                first_step_multivector_query=FIRST_STEP_MULTIVECTOR_QUERY
            )

        """
        facets = [] if facets is None else facets
        filters = [] if filters is None else filters

        return self.services.search.multistep_chunk(
            dataset_id=self.dataset_id,
            multivector_query=multivector_query,
            first_step_multivector_query=first_step_multivector_query,
            chunk_field=chunk_field,
            chunk_scoring=chunk_scoring,
            chunk_page_size=chunk_page_size,
            chunk_page=chunk_page,
            approximation_depth=approximation_depth,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            facets=facets,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
            asc=asc,
            keep_search_history=keep_search_history,
            hundred_scale=hundred_scale,
            first_step_page=first_step_page,
            first_step_page_size=first_step_page_size,
            query=query,
        )
