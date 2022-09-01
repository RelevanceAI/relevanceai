import traceback
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy

from typing import Optional, Union, Callable, Dict, Any, Set, List

from relevanceai.utils.decorators.analytics import track

from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.cluster.transform import ClusterTransform

from relevanceai.constants import Warning
from relevanceai.constants.errors import MissingClusterError
from relevanceai.constants import MissingClusterError, Warning


class ClusterOps(ClusterTransform, OperationAPIBase):
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
        model=None,
        model_kwargs=None,
        cluster_field: str = "_cluster_",
        byo_cluster_field: str = None,
        include_cluster_report: bool = False,
        verbose: bool = False,
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

        self.include_cluster_report = include_cluster_report

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
            include_cluster_report=include_cluster_report,
            **kwargs,
        )

        # alias is set after model so that we can get the number of clusters
        # if the model needs to be instantiated
        self.alias = self._get_alias(alias)

        self.byo_cluster_field = byo_cluster_field
        if byo_cluster_field is not None:
            self.create_byo_clusters()

    def post_run(self, dataset, documents, updated_documents):
        centroid_documents = self.get_centroid_documents()
        self.insert_centroids(centroid_documents)
        if hasattr(self, "include_cluster_report") and self.include_cluster_report:
            try:
                from relevanceai.recipes.model_observability.cluster.report import (
                    ClusterReport,
                )

                app = ClusterReport(f"Cluster Report for {self.alias}", dataset)
                app.start_cluster_evaluator(
                    self.get_field_across_documents(self.vector_fields[0], documents),
                    self.get_field_across_documents(
                        self._get_cluster_field_name(), updated_documents
                    ),
                    # centroids=centroid_documents
                )
                app.evaluator.X_silhouette_samples = np.array(
                    self.get_field_across_documents(
                        self._silhouette_score_field_name(), updated_documents
                    )
                )
                app.evaluator.X_squared_error_samples = np.array(
                    self.get_field_across_documents(
                        self._squared_error_field_name(), updated_documents
                    )
                )
                app.section_cluster_report()
                print()
                print("We've built your cluster report app:")
                app.deploy()
            except Exception as e:
                print(e)
                print("Couldnt' create cluster report.")
        return

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
                centorid_documents=[
                    {"_id" : "cluster-0", "sample_1_vector_": [1, 1, 1]},
                    {"_id" : "cluster-1", "sample_1_vector_": [1, 2, 2]},
                ]
            )

        """
        res = self.datasets.cluster.centroids.insert(
            dataset_id=self.dataset_id,
            cluster_centers=self.json_encoder(centroid_documents),
            vector_fields=self.vector_fields,
            alias=self.alias,
        )
        return res

    def calculate_centroids(self, method="mean"):
        """
        calculates the centroids from the dataset vectors
        """

        # calculate the centroids
        centroid_vectors = {}

        def calculate_centroid(vectors):
            X = np.array(vectors)
            return X.mean(axis=0)

        centroid_vectors = self._operate_across_clusters(
            field=self.vector_fields[0], func=calculate_centroid
        )

        if isinstance(centroid_vectors, dict):
            centroid_vectors = [
                {"_id": k, self.vector_fields[0]: v}
                for k, v in centroid_vectors.items()
            ]
        return centroid_vectors

    def create_centroids(self, insert: bool = True):
        """
        Calculate centroids from your dataset vectors.

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

        if insert:
            self.insert_centroids(
                centroid_documents=centroid_vectors,
            )
        return centroid_vectors

    def get_centroid_documents(self):
        centroid_vectors = {}
        if hasattr(self.model, "_centroids") and self.model._centroids is not None:
            # TODO: fix this so that it creates the proper labels
            centroid_vectors = self.model._centroids.tolist()
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

        raise ValueError(f"Missing the centroid with id {cluster_id}")

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
        if alias is None:
            alias = self.alias
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

    def merge(self, target_cluster_id: str, cluster_ids: list):
        """
        Merge clusters into the target cluster.
        The centroids are re-calculated and become a new middle.
        """
        return self.datasets.cluster.merge(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            cluster_ids=[target_cluster_id] + cluster_ids,
        )

    def create_byo_clusters(self):
        """
        Create BYO clusters for a given field
        """
        # TODO: Change into generator to make unique values more than 9999
        results = self.datasets.facets(
            dataset_id=self.dataset_id, fields=[self.byo_cluster_field], page_size=9999
        )

        try:
            for r in results["results"][self.byo_cluster_field]:
                filters = [
                    {
                        "field": self.byo_cluster_field,
                        "filter_type": "exact_match",
                        "condition": "==",
                        "condition_value": r["value"],
                    }
                ]
                cluster_doc = {}
                self.set_field(self._get_cluster_field_name(), cluster_doc, r["value"])
                results = self.datasets.documents.update_where(
                    dataset_id=self.dataset_id, update=cluster_doc, filters=filters
                )

        except KeyError:
            raise ValueError("Cluster field has no values.")

        return results

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

    @property
    def labels(self):
        metadata = self.datasets.metadata(self.dataset_id)
        metadata = metadata.get("results", {}).get("cluster_metadata", {})
        vector_fields = ["text_mpnet_vector_"]
        alias = "kmeans-100"

        cluster_field = "_cluster_." + ".".join(vector_fields) + "." + alias
        labels = metadata.get("labels", {}).get(cluster_field, {})
        print("To view nicely, please use `pd.DataFrame(labels)`.")
        return labels

    def create_parent_cluster(
        self,
        to_merge: dict,
        new_cluster_field: str,
    ):
        """
        to_merge should look similar to below:

        .. code-block::

            to_merge = {
                0: [
                    'cluster_1',
                    'cluster_2'
                ]
            }

        """
        parent_field = self._get_cluster_field_name()
        for cluster, clusters_to_combine in to_merge.items():
            for i, cluster_to_combine in enumerate(clusters_to_combine):
                # Update the documents in the cluster to become a subcluster
                cluster_filter = [
                    {
                        "field": parent_field,
                        "filter_type": "contains",
                        "condition": "==",
                        "condition_value": cluster_to_combine,
                    }
                ]

                if "cluster_" in cluster_to_combine:
                    cluster_to_combine = cluster_to_combine.replace("cluster_", "")
                if isinstance(cluster, int):
                    update = {
                        new_cluster_field: f"mergedCluster_{cluster}-{cluster_to_combine}"
                    }
                elif isinstance(cluster, str):
                    update = {new_cluster_field: f"{cluster}-{cluster_to_combine}"}
                updated = self.datasets.documents.update_where(
                    dataset_id=self.dataset_id, update=update, filters=cluster_filter
                )
                print("Update status: ")
                print(updated)

            # Merge the original clusters combine to create a subcluster
            if isinstance(cluster, int):
                try:
                    merge_results = self.merge(
                        target_cluster_id=clusters_to_combine[0],
                        # target_cluster_id=f"mergedCluster_{cluster}",
                        cluster_ids=clusters_to_combine[1:],
                    )
                    print(merge_results)
                except Exception as e:
                    print(e)
            elif isinstance(cluster, str):
                self.merge(target_cluster_id=cluster, cluster_ids=clusters_to_combine)

        self.append_metadata_list(
            field="_subcluster_",
            value_to_append={
                "parent_field": parent_field,
                "cluster_field": new_cluster_field,
            },
            only_unique=True,
        )

        # Port over the labels from the cluster to the subcluster
        metadata = self.datasets.metadata(self.dataset_id)["results"]
        if parent_field in metadata["cluster_metadata"]["labels"]:
            for k, old_labels in to_merge.items():
                for l in old_labels:
                    label = metadata["cluster_metadata"]["labels"][parent_field][
                        "labels"
                    ][l]
                    if new_cluster_field not in metadata["cluster_metadata"]["labels"]:
                        metadata["cluster_metadata"]["labels"].update(
                            {new_cluster_field: {"labels": {}}}
                        )
                    l = l.replace("cluster_", "")
                    cluster_id = f"mergedCluster_{k}-{l}"
                    metadata["cluster_metadata"]["labels"][new_cluster_field]["labels"][
                        cluster_id
                    ] = label
            results = self.datasets.post_metadata(self.dataset_id, metadata)
            print("Updated metadata")
            print(results)

    def explain_text_clusters(
        self,
        text_field,
        encode_fn_or_model,
        n_closest: int = 5,
        highlight_output_field="_explain_",
        algorithm: str = "relational",
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
            from relevanceai.operations_new.vectorize.text.transform import (
                VectorizeTextTransform,
            )

            self.model = VectorizeTextTransform._get_model(encode_fn_or_model)
            encode_fn = self.model.encode
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

    @track
    def aggregate(
        self,
        metrics: Optional[list] = None,
        sort: Optional[list] = None,
        groupby: Optional[list] = None,
        filters: Optional[list] = None,
        page_size: int = 20,
        page: int = 1,
        asc: bool = False,
        flatten: bool = True,
    ):
        """
        Takes an aggregation query and gets the aggregate of each cluster in a collection. This helps you interpret each cluster and what is in them.
        It can only can be used after a vector field has been clustered. \n
        Aggregation/Groupby of a collection using an aggregation query. The aggregation query is a json body that follows the schema of:

        .. code-block::

            {
                "groupby" : [
                    {"name": <alias>, "field": <field in the collection>, "agg": "category"},
                    {"name": <alias>, "field": <another groupby field in the collection>, "agg": "numeric"}
                ],
                "metrics" : [
                    {"name": <alias>, "field": <numeric field in the collection>, "agg": "avg"}
                    {"name": <alias>, "field": <another numeric field in the collection>, "agg": "max"}
                ]
            }

        For example, one can use the following aggregations to group score based on region and player name.

        .. code-block::

            {
                "groupby" : [
                    {"name": "region", "field": "player_region", "agg": "category"},
                    {"name": "player_name", "field": "name", "agg": "category"}
                ],
                "metrics" : [
                    {"name": "average_score", "field": "final_score", "agg": "avg"},
                    {"name": "max_score", "field": "final_score", "agg": "max"},
                    {'name':'total_score','field':"final_score", 'agg':'sum'},
                    {'name':'average_deaths','field':"final_deaths", 'agg':'avg'},
                    {'name':'highest_deaths','field':"final_deaths", 'agg':'max'},
                ]
            }
        "groupby" is the fields you want to split the data into. These are the available groupby types:
            - category : groupby a field that is a category
            - numeric: groupby a field that is a numeric
        "metrics" is the fields and metrics you want to calculate in each of those, every aggregation includes a frequency metric. These are the available metric types:
            - "avg", "max", "min", "sum", "cardinality"
        The response returned has the following in descending order. \n
        If you want to return documents, specify a "group_size" parameter and a "select_fields" parameter if you want to limit the specific fields chosen. This looks as such:
            .. code-block::
                {
                    'groupby':[
                        {'name':'Manufacturer','field':'manufacturer','agg':'category',
                        'group_size': 10, 'select_fields': ["name"]},
                    ],
                    'metrics':[
                        {'name':'Price Average','field':'price','agg':'avg'},
                    ],
                }
                # ouptut example:
                {"title": {"title": "books", "frequency": 200, "documents": [{...}, {...}]}, {"title": "books", "frequency": 100, "documents": [{...}, {...}]}}
        For array-aggregations, you can add "agg": "array" into the aggregation query.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        metrics: list
            Fields and metrics you want to calculate
        groupby: list
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten
        alias: string
            Alias used to name a vector field. Belongs in field_{alias} vector
        metrics: list
            Fields and metrics you want to calculate
        groupby: list
            Fields you want to split the data into
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten: bool
            Whether to flatten

        Example
        ---------
        .. code-block::
            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
            cluster_ops.run(df, vector_fields=["sample_vector_"])
            clusterer.aggregate(
                "sample_dataset_id",
                groupby=[{
                    "field": "title",
                    "agg": "wordcloud",
                }],
                vector_fields=['sample_vector_']
            )
        """
        metrics = [] if metrics is None else metrics
        groupby = [] if groupby is None else groupby
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort

        return self.datasets.cluster.aggregate(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            groupby=groupby,
            metrics=metrics,
            sort=sort,
            filters=filters,
            alias=self.alias,
            page_size=page_size,
            page=page,
            asc=asc,
            flatten=flatten,
        )
