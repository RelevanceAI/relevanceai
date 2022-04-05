from typing import List, Dict, Optional, Any, Union

from relevanceai.client.helpers import Credentials
from relevanceai.utils.decorators import deprecated, beta
from relevanceai._api import APIClient


class Operations(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
    ):
        self.credentials = credentials
        self.dataset_id = dataset_id
        super().__init__(self.credentials)

    def cluster(
        self,
        model: Union[str, Any],
        vector_fields: List[str],
        alias: Optional[str] = None,
        **kwargs,
    ):
        """
        Run clustering on your dataset.

        Example
        ----------

        .. code-block::

            from sklearn.cluster import KMeans
            model = KMeans()

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            cluster_ops = ds.cluster(
                model=model, vector_fields=["sample_vector_"],
                alias="kmeans-8"
            )

        Parameters
        ------------

        model: Union[str, Any]
            Any K-Means model
        vector_fields: List[str]
            A list of possible vector fields
        alias: str
            The alias to be used to store your model

        """
        from relevanceai.operations.cluster import ClusterOps

        ops = ClusterOps(
            credentials=self.credentials,
            model=model,
            alias=alias,
            vector_fields=vector_fields,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops(dataset_id=self.dataset_id, vector_fields=vector_fields)

    def reduce_dims(
        self,
        model: Any,
        n_components: int,
        alias: str,
        vector_fields: List[str],
        **kwargs,
    ):
        """
        Reduce dimensions

        Parameters
        --------------

        model: Callable
            model to reduce dimensions
        n_components: int
            The number of components
        alias: str
            The alias of the model
        vector_fields: List[str]
            The list of vector fields to support

        """
        from relevanceai.operations.dr import ReduceDimensionsOps

        ops = ReduceDimensionsOps(
            credentials=self.credentials,
            model=model,
            n_components=n_components,
            **kwargs,
        )
        return ops.operate(
            dataset_id=self.dataset_id,
            vector_fields=vector_fields,
            alias=alias,
        )

    def vectorize(
        self,
        text_fields=None,
        image_fields=None,
        **kwargs,
    ):
        """
        Vectorize the model

        Parameters
        ----------
        image_fields: List[str]
            A list of image fields to vectorize

        text_fields: List[str]
            A list of text fields to vectorize

        image_encoder
            A deep learning image encoder from the vectorhub library. If no
            encoder is specified, a default encoder (Clip2Vec) is loaded.

        text_encoder
            A deep learning text encoder from the vectorhub library. If no
            encoder is specified, a default encoder (USE2Vec) is loaded.

        Returns
        -------
        dict
            If the vectorization process is successful, this dict contains
            the added vector names. Else, the dict is the request result
            containing error information.

        Example
        -------
        .. code-block::

            from relevanceai import Client
            from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec

            text_model = SentenceTransformer2Vec("all-mpnet-base-v2 ")

            client = Client()

            dataset_id = "sample_dataset_id"
            ds = client.Dataset(dataset_id)

            ds.vectorize(
                image_fields=["image_field_1", "image_field_2"],
                text_fields=["text_field"],
                text_model=text_model
            )


        """

        from relevanceai.operations.vector import VectorizeOps

        ops = VectorizeOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
            **kwargs,
        )
        return ops.vectorize(
            text_fields=text_fields,
            image_fields=image_fields,
        )

    def vector_search(self, **kwargs):
        """
        Allows you to leverage vector similarity search to create a semantic search engine. Powerful features of VecDB vector search:

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
            Whether to store the history into VecDB. This will increase the storage costs over time.
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
            ds = client.Dataset("sample")
            results = ds.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """

        from relevanceai.operations.vector import SearchOps

        ops = SearchOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.vector_search(**kwargs)

    def hybrid_search(self, **kwargs):
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


        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            MULTIVECTOR_QUERY = [{"vector": [0, 1, 2], "fields": ["sample_vector_"]}]
            results = ds.vector_search(multivector_query=MULTIVECTOR_QUERY)

        """
        from relevanceai.operations.vector import SearchOps

        ops = SearchOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.hybrid_search(**kwargs)

    def chunk_search(self, **kwargs):
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
            Whether to store the history into VecDB. This will increase the storage costs over time.
        hundred_scale: bool
            Whether to scale up the metric by 100
        query: string
            What to store as the query name in the dashboard

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            results = ds.chunk_search(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY
            )

        """
        from relevanceai.operations.vector import SearchOps

        ops = SearchOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )

        return ops.chunk_search(**kwargs)

    def multistep_chunk_search(self, **kwargs):
        """
        Multistep chunk search involves a vector search followed by chunk search, used to accelerate chunk searches or to identify context before delving into relevant chunks. e.g. Search against the paragraph vector first then sentence chunkvector after. \n

        For more information about chunk search check out services.search.chunk. \n

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

        Example
        -----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            results = ds.search.multistep_chunk(
                chunk_field="_chunk_",
                multivector_query=MULTIVECTOR_QUERY,
                first_step_multivector_query=FIRST_STEP_MULTIVECTOR_QUERY
            )

        """
        from relevanceai.operations.vector import SearchOps

        ops = SearchOps(
            credentials=self.credentials,
            dataset_id=self.dataset_id,
        )
        return ops.multistep_chunk_search(**kwargs)

    def launch_cluster_app(self, configuration: dict = None):
        """
        Launch an app with a given configuration


        Example
        --------

        .. code-block::

            ds.launch_cluster_app()

        Parameters
        -----------

        configuration: dict
            The configuration can be found in the deployable once created.

        """
        if configuration is None:
            url = f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            print(
                "Build your clustering app here: "
                f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
            return
        if "configuration" in configuration:
            configuration = configuration["configuration"]
        results = self.deployables.create(
            dataset_id=self.dataset_id, configuration=configuration
        )

        # After you have created an app
        url = f"https://cloud.relevance.ai/dataset/{results['dataset_id']}/deploy/cluster/{self.project}/{self.api_key}/{results['deployable_id']}/{self.region}"
        print(f"You can now access your deployable at {url}.")
        return url

    def subcluster(self, model, alias: str, vector_fields, parent_field, **kwargs):
        """
        Subcluster
        """
        from relevanceai.operations.cluster import SubClusterOps

        ops = SubClusterOps(
            model=model,
            credentials=self.credentials,
            alias=alias,
            vector_fields=vector_fields,
            dataset_id=self.dataset_id,
            parent_field=parent_field,
            dataset=self.dataset_id,
            **kwargs,
        )
        return ops.fit_predict(dataset=self.dataset_id, vector_fields=vector_fields)

    def add_sentiment(
        self,
        field: str,
        output_field: str = None,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        log_to_file: bool = True,
        chunksize: int = 20,
        workflow_alias: str = "sentiment",
        notes=None,
    ):
        """
        Easily add sentiment to your dataset

        Example
        ----------

        .. code-block::

            ds.add_sentiment(field="sample_1_label")

        Parameters
        --------------

        field: str
            The field to add sentiment to
        output_field: str
            Where to store the sentiment values
        model_name: str
            The HuggingFace Model name.
        log_to_file: bool
            If True, puts the logs in a file. Otherwise, it will

        """
        from relevanceai.operations.text.sentiment.sentiment_workflow import (
            SentimentWorkflow,
        )

        if output_field is None:
            output_field = "_sentiment_." + field
        workflow = SentimentWorkflow(
            model_name=model_name, workflow_alias=workflow_alias
        )
        return workflow.fit_dataset(
            dataset=self,
            input_field=field,
            output_field=output_field,
            log_to_file=log_to_file,
            chunksize=chunksize,
            workflow_alias=workflow_alias,
            notes=notes,
        )
