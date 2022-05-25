import warnings

from copy import deepcopy

from typing import Optional, Union, Callable, Dict, Any, Set, List

from relevanceai.utils.decorators.analytics import track

from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.cluster.alias import ClusterAlias
from relevanceai.operations_new.cluster.base import ClusterBase

from relevanceai.constants import Warning
from relevanceai.constants.errors import MissingClusterError
from relevanceai.constants import MissingClusterError, Warning


class ClusterOps(ClusterBase, OperationAPIBase, ClusterAlias):
    """
    Cluster-related functionalities
    """

    # These need to be instantiated on __init__
    model_name: str

    def __init__(
        self,
        dataset_id: str,
        vector_fields: list,
        alias: str,
        cluster_field: str = "_cluster_",
        verbose: bool = False,
        model=None,
        model_kwargs=None,
        *args,
        **kwargs,
    ):
        """
        ClusterOps objects
        """
        self.dataset_id = dataset_id
        self.vector_fields = vector_fields
        self.cluster_field = cluster_field
        self.verbose = verbose
        self.model = model
        if isinstance(self.model, str):
            self.model_name = self.model
        else:
            self.model_name = str(self.model)

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        super().__init__(
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            alias=alias,
            cluster_field=cluster_field,
            verbose=verbose,
            model=model,
            model_kwargs=model_kwargs,
            **kwargs,
        )

        # alias is set after model so that we can get the number of clusters
        # if the model needs ot be instantiated
        self.alias = self._get_alias(alias)

    def _operate(self, cluster_id: str, field: str, output: dict, func: Callable):
        """
        Internal function for operations

        It takes a cluster_id, a field, an output dictionary, and a function, and then it gets all the
        documents in the cluster, gets the field across all the documents, and then applies the function
        to the field

        Parameters
        ----------
        cluster_id : str
            str, field: str, output: dict, func: Callable
        field : str
            the field you want to get the value for
        output : dict
            dict
        func : Callable
            Callable

        """
        cluster_field = self._get_cluster_field_name()
        # TODO; change this to fetch all documents
        documents = self._get_all_documents(
            self.dataset_id,
            filters=[
                {
                    "field": cluster_field,
                    "filter_type": "exact_match",
                    "condition": "==",
                    "condition_value": cluster_id,
                },
                {
                    "field": field,
                    "filter_type": "exists",
                    "condition": ">=",
                    "condition_value": " ",
                },
            ],
            select_fields=[field, cluster_field],
            show_progress_bar=False,
        )
        # get the field across each
        arr = self.get_field_across_documents(field, documents)
        output[cluster_id] = func(arr)

    def _operate_across_clusters(self, field: str, func: Callable):
        output: Dict[str, Any] = dict()
        for cluster_id in self.list_cluster_ids():
            self._operate(cluster_id=cluster_id, field=field, output=output, func=func)
        return output

    def list_cluster_ids(
        self,
        alias: str = None,
        minimum_cluster_size: int = 0,
        num_clusters: int = 1000,
    ):
        """
        List unique cluster IDS

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            cluster_ops = client.ClusterOps(
                alias="kmeans_8", vector_fields=["sample_vector_]
            )
            cluster_ops.list_cluster_ids()

        Parameters
        -------------
        alias: str
            The alias to use for clustering
        minimum_cluster_size: int
            The minimum size of the clusters
        num_clusters: int
            The number of clusters

        """
        # Mainly to be used for subclustering
        # Get the cluster alias
        cluster_field = self._get_cluster_field_name()

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=self.dataset_id,
                fields=[cluster_field],
                page_size=int(self.config["data.max_clusters"]),
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if cluster_field not in facet_results:
                raise MissingClusterError(alias=alias)
            for facet in facet_results[cluster_field]:
                if facet["frequency"] > minimum_cluster_size:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[cluster_field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)

        return list(all_cluster_ids)

    def insert_centroids(
        self,
        centroid_documents,
    ) -> None:
        """
        Insert centroids
        Centroids look below

        .. code-block::

            cluster_ops = client.ClusterOps(
                vector_field=["sample_1_vector_"],
                alias="sample"
            )
            cluster_ops.insert_centroids(
                centorid_documents={
                    "cluster-0": [1, 1, 1],
                    "cluster-2": [2, 1, 1]
                }
            )

        """
        # Centroid documents are in the format {"cluster-0": [1, 1, 1]}
        return self.datasets.cluster.centroids.insert(
            dataset_id=self.dataset_id,
            cluster_centers=self.json_encoder(centroid_documents),
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    def create_centroids(self):
        """
        Calculate centroids from your vectors

        Example
        --------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")
            cluster_ops = ds.ClusterOps(
                alias="kmeans-25",
                vector_fields=['sample_vector_']
            )
            centroids = cluster_ops.create_centroids()

        """
        # Get an array of the different vectors
        if len(self.vector_fields) > 1:
            raise NotImplementedError(
                "Do not currently support multiple vector fields for centroid creation."
            )

        # calculate the centroids
        centroid_vectors = self.calculate_centroids()

        self.insert_centroids(
            centroid_documents=centroid_vectors,
        )
        return centroid_vectors

    def calculate_centroids(self):
        import numpy as np

        # calculate the centroids
        centroid_vectors = {}

        def calculate_centroid(vectors):
            X = np.array(vectors)
            return X.mean(axis=0)

        centroid_vectors = self._operate_across_clusters(
            field=self.vector_fields[0], func=calculate_centroid
        )

        # Does this insert properly?
        if isinstance(centroid_vectors, dict):
            centroid_vectors = [
                {"_id": k, self.vector_fields[0]: v}
                for k, v in centroid_vectors.items()
            ]
        return centroid_vectors

    def get_centroid_documents(self):
        centroid_vectors = {}
        if self.model._centroids is not None:
            centroid_vectors = self.model._centroids
            # get the cluster label function
            labels = range(len(centroid_vectors))
            cluster_ids = self.format_cluster_labels(labels)
            if len(self.vector_fields) > 1:
                warnings.warn(
                    "Currently do not support inserting centroids with multiple vector fields"
                )
            centroids = [
                {"_id": k, self.vector_fields[0]: v}
                for k, v in zip(cluster_ids, centroid_vectors)
            ]
        else:
            centroids = self.create_centroids()

        return centroids

    def list_closest(
        self,
        cluster_ids: Optional[list] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[list] = None,
        facets: Optional[list] = None,
        include_vector: bool = False,
        cluster_properties_filters: Optional[Dict] = None,
        include_count: bool = False,
        include_facets: bool = False,
        verbose: bool = False,
    ):
        """
        List of documents closest from the center.
        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_fields: list
            The vector fields where a clustering task runs
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        centroid_vector_fields: list
            Vector fields stored
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results
        cluster_properties_filter: dict
            Filter if clusters with certain characteristics should be hidden in results
        """
        if cluster_properties_filters is None:
            cluster_properties_filters = {}
        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
            select_fields=select_fields,
            approx=approx,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            filters=filters,
            facets=facets,
            include_vector=include_vector,
            include_count=include_count,
            include_facets=include_facets,
            cluster_properties_filter=cluster_properties_filters,
            verbose=verbose,
        )

    @track
    def list_furthest(
        self,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 3,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        # facets: List = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
        cluster_properties_filter: Optional[Dict] = {},
    ):
        """
        List documents furthest from the center.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_fields: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        select_fields: list
            Fields to include in the search results, empty array/list means all fields
        approx: int
            Used for approximate search to speed up search. The higher the number, faster the search but potentially less accurate
        sum_fields: bool
            Whether to sum the multiple vectors similarity search score as 1 or seperate
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        similarity_metric: string
            Similarity Metric, choose from ['cosine', 'l1', 'l2', 'dp']
        filters: list
            Query for filtering the search results
        facets: list
            Fields to include in the facets, if [] then all
        min_score: int
            Minimum score for similarity metric
        include_vectors: bool
            Include vectors in the search results
        include_count: bool
            Include the total count of results in the search results
        include_facets: bool
            Include facets in the search results
        """
        return self.datasets.cluster.centroids.list_furthest_from_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
            centroid_vector_fields=centroid_vector_fields,
            select_fields=select_fields,
            approx=approx,
            sum_fields=sum_fields,
            page_size=page_size,
            page=page,
            similarity_metric=similarity_metric,
            filters=filters,
            min_score=min_score,
            include_vector=include_vector,
            include_count=include_count,
            cluster_properties_filter=cluster_properties_filter,
        )

    def explain_text_clusters(
        self,
        text_field,
        encode_fn_or_model,
        n_closest: int = 5,
        highlight_output_field="_explain_",
        algorithm: str = "centroid",
        model_kwargs: Optional[dict] = None,
    ):
        """
        It takes a text field and a function that encodes the text field into a vector.
        It then returns the top n closest vectors to each cluster centroid.
        .. code-block::
            def encode(X):
                return [1, 2, 1]
            cluster_ops.explain_text_clusters(text_field="hey", encode_fn_or_model=encode)

        Parameters
        ----------
        text_field
            The field in the dataset that contains the text to be explained.
        encode_fn
            This is the function that will be used to encode the text.
        n_closest : int, optional
            The number of closest documents to each cluster to return.
        highlight_output_field, optional
            The name of the field that will be added to the output dataset.
        algorithm: str
            Algorithm is either "centroid" or "relational"

        Returns
        -------
            A new dataset with the same data as the original dataset, but with a new field called _explain_
        """
        if isinstance(encode_fn_or_model, str):
            # Get the model
            raise NotImplementedError(
                "Model strings not supported yet. Please supply a function."
            )
            # model_kwargs = {} if model_kwargs is None else model_kwargs
            # self.vectorizer = self._get_model(encode_fn_or_model, model_kwargs)
            # if hasattr(self.vectorizer, "encode"):
            #     encode_fn = self.vectorizer.encode
            # else:
            #     raise AttributeError("Vectorizer is missing an `encode` function.")
        else:
            encode_fn = encode_fn_or_model

        from relevanceai.operations_new.cluster.text.explainer.ops import (
            TextClusterExplainerOps,
        )

        ops = TextClusterExplainerOps(credentials=self.credentials)
        if algorithm == "centroid":
            return ops.explain_clusters(
                dataset_id=self.dataset_id,
                alias=self.alias,
                vector_fields=self.vector_fields,
                text_field=text_field,
                encode_fn=encode_fn,
                n_closest=n_closest,
                highlight_output_field=highlight_output_field,
            )
        elif algorithm == "relational":
            return ops.explain_clusters_relational(
                dataset_id=self.dataset_id,
                alias=self.alias,
                vector_fields=self.vector_fields,
                text_field=text_field,
                encode_fn=encode_fn,
                n_closest=n_closest,
                highlight_output_field=highlight_output_field,
            )
        raise ValueError("Algorithm needs to be either `relational` or `centroid`.")

    def store_operation_metadatas(self):
        self.store_operation_metadata(
            operation="cluster",
            values=str(
                {
                    "model": self.model,
                    "vector_fields": self.vector_fields,
                    "alias": self.alias,
                    "model_kwargs": self.model_kwargs,
                }
            ),
        )

    @property
    def centroids(self):
        """
        Access the centroids of your dataset easily

        .. code-block::

            ds = client.Dataset("sample")
            cluster_ops = ds.ClusterOps(
                vector_fields=["sample_vector_"],
                alias="simple"
            )
            cluster_ops.centroids

        """
        if not hasattr(self, "_centroids"):
            self._centroids = self.datasets.cluster.centroids.documents(
                dataset_id=self.dataset_id,
                vector_fields=self.vector_fields,
                alias=self.alias,
                page_size=9999,
                include_vector=True,
            )["results"]
        return self._centroids

    def get_centroid_from_id(
        self,
        cluster_id: str,
    ) -> Dict[str, Any]:
        """> It takes a cluster id and returns the centroid with that id

        Parameters
        ----------
        cluster_id : str
            The id of the cluster to get the centroid for.

        Returns
        -------
            The centroid with the given id.

        """

        for centroid in self.centroids:
            if centroid["_id"] == cluster_id:
                return centroid

        raise ValueError(f"Missing the centorid with id {cluster_id}")

    @staticmethod
    def _get_filters(
        filters: List[Dict[str, Union[str, int]]],
        vector_fields: List[str],
    ) -> List[Dict[str, Union[str, int]]]:
        """It takes a list of filters and a list of vector fields and returns a list of filters that
        includes the original filters and a filter for each vector field that checks if the vector field
        exists

        Parameters
        ----------
        filters : List[Dict[str, Union[str, int]]]
            List[Dict[str, Union[str, int]]]
        vector_fields : List[str]
            List[str] = ["vector_field_1", "vector_field_2"]

        Returns
        -------
            A list of dictionaries.

        """

        vector_field_filters = [
            {
                "field": vector_field,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for vector_field in vector_fields
        ]

        filters = deepcopy(filters)

        if filters is None:
            filters = vector_field_filters
        else:
            filters += vector_field_filters  # type: ignore

        return filters

    def merge(self, cluster_ids: list):
        """
        Merge clusters into the first one.
        The centroids are re-calculated and become a new middle.
        """
        return self.datasets.cluster.merge(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=cluster_ids,
        )
