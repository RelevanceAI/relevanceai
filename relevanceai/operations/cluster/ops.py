"""
ClusterOps class to run clustering. It is intended to be integrated with
models that inherit from `ClusterBase`.

You can run the ClusterOps as such:

.. code-block::

    from relevanceai import Client
    client = Client()
    df = client.Dataset("sample_dataset")

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=2)
    cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
    cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

You can view other examples of how to interact with this class here :ref:`integration`.

"""
import numpy as np

from tqdm.auto import tqdm

from typing import Union, Optional, Callable, Set, List, Dict, Any

from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.decorators.version import beta

from relevanceai.operations.cluster.partial import PartialClusterOps
from relevanceai.operations.cluster.sub import SubClusterOps
from relevanceai.operations.cluster.groupby import ClusterGroupby, ClusterAgg
from relevanceai.operations.cluster.constants import METRIC_DESCRIPTION
from relevanceai.operations.cluster.base import (
    ClusterBase,
    CentroidClusterBase,
    BatchClusterBase,
)
from relevanceai.reports.cluster import ClusterReport
from relevanceai.constants.errors import NoDocumentsError, NoModelError


class ClusterOps(PartialClusterOps, SubClusterOps):
    """
    ClusterOps class allows users to set up any clustering model to fit on a Dataset.

    You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Example
    -----------

    .. code-block::

        from relevanceai import Client
        client = Client()
        df = client.Dataset("sample_dataset")

        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=2)
        cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
        cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

    """

    def __init__(
        self,
        project: str,
        api_key: str,
        firebase_uid: str,
        model: Union[BatchClusterBase, ClusterBase, CentroidClusterBase, Any] = None,
        alias: Optional[str] = None,
        cluster_field: str = "_cluster_",
        parent_alias: str = None,
    ):
        self.alias = alias  # type: ignore
        self.parent_alias = parent_alias
        self.cluster_field = cluster_field
        if model is None:
            raise NoModelError

        self.model = self._assign_model(model)

        self.firebase_uid = firebase_uid

        if project is None or api_key is None:
            project, api_key = self._token_to_auth()
        else:
            self.project: str = project
            self.api_key: str = api_key

        self.verbose = True

        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

    def __call__(self):
        self.groupby = ClusterGroupby(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )
        self.agg = ClusterAgg(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    @track
    def groupby(self):
        return ClusterGroupby(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )

    @track
    def agg(self):
        self.groupby = ClusterGroupby(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            alias=self.alias,
            vector_fields=self.vector_fields,
        )
        return ClusterAgg(
            project=self.project,
            api_key=self.api_key,
            dataset_id=self.dataset_id,
            firebase_uid=self.firebase_uid,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    @track
    def list_closest_to_center(
        self,
        vector_fields: Optional[List] = None,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
        select_fields: Optional[List] = None,
        approx: int = 0,
        sum_fields: bool = True,
        page_size: int = 1,
        page: int = 1,
        similarity_metric: str = "cosine",
        filters: Optional[List] = None,
        # facets: List = [],
        min_score: int = 0,
        include_vector: bool = False,
        include_count: bool = True,
    ):
        """
        List of documents closest from the centre.

        Parameters
        ----------

        cluster_ids: list
            Any of the cluster ids
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

        Example
        --------------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
            cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

            cluster_ops.list_closest_to_center()

        """
        cluster_ids = [] if cluster_ids is None else cluster_ids
        centroid_vector_fields = (
            [] if centroid_vector_fields is None else centroid_vector_fields
        )
        select_fields = [] if select_fields is None else select_fields
        filters = [] if filters is None else filters

        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields
            if vector_fields is None
            else vector_fields,
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
        )

    @track
    def aggregate(
        self,
        vector_fields: List[str] = None,
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

        Parameters
        ----------
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
            cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

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
        sort = [] if sort is None else sort
        groupby = [] if groupby is None else groupby
        filters = [] if filters is None else filters

        return self.services.cluster.aggregate(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields if not vector_fields else vector_fields,
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

    def list_furthest_from_center(self, vector_fields: list = None):
        """
        List of documents furthest from the centre.

        Parameters
        ----------
        cluster_ids: list
            Any of the cluster ids
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

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
            cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

            cluster_ops.list_furthest_from_center()

        """
        return self.datasets.cluster.centroids.list_furthest_from_center(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields
            if vector_fields is None
            else vector_fields,
            alias=self.alias,
        )

    @track
    def insert_centroid_documents(self, centroid_documents: List[Dict]):
        """
        Insert the centroid documents

        Parameters
        ------------

        centroid_documents: List[Dict]
            Insert centroid documents
        dataset_id: str
            Dataset to insert

        Example
        ------------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)
            cluster_ops.fit_predict_update(df, vector_fields=["sample_vector_"])

            centroids = cluster_ops.get_centroid_documents()
            cluster_ops.insert_centroid_documents(centroids)

        """

        results = self.services.cluster.centroids.insert(
            dataset_id=self.dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )
        return results

    @track
    def get_centroid_documents(self) -> List:
        """
        Get the centroid documents to store. This enables you to use `list_closest_to_center()`
        and `list_furthest_from_center`.

        .. code-block::

            {
                "_id": "document-id-1",
                "centroid_vector_": [0.23, 0.24, 0.23]
            }

        If multiple vector fields returns this:
        Returns multiple

        .. code-block::

            {
                "_id": "document-id-1",
                "blue_vector_": [0.12, 0.312, 0.42],
                "red_vector_": [0.23, 0.41, 0.3]
            }

        """
        if hasattr(self.model, "get_centroid_documents"):
            self.model.vector_fields = self.vector_fields
            return self.model.get_centroid_documents()
        self.centers = self.model.get_centers()

        if not hasattr(self, "vector_fields") or len(self.vector_fields) == 1:
            if isinstance(self.centers, np.ndarray):
                self.centers = self.centers.tolist()
            centroid_vector_field_name = self.vector_fields[0]
            return [
                {
                    "_id": self._label_cluster(i),
                    centroid_vector_field_name: self.centers[i],
                }
                for i in range(len(self.centers))
            ]
        # For one or more vectors, separate out the vector fields
        # centroid documents are created using multiple vector fields
        centroid_docs = []
        for i, c in enumerate(self.centers):
            centroid_doc = {"_id": self._label_cluster(i)}
            for j, vf in enumerate(self.vector_fields):
                centroid_doc[vf] = self.centers[i][vf]
            centroid_docs.append(centroid_doc.copy())
        return centroid_docs

    @property
    def centroids(self):
        """
        See your centroids if there are any.
        """
        return self.services.cluster.centroids.documents(
            self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            page_size=10000,
            # cursor: str = None,
            include_vector=True,
        )

    @track
    def delete_centroids(self, dataset_id: str, vector_fields: List):
        """Delete the centroids after clustering."""
        # TODO: Fix delete centroids once its moved over to Node JS
        import requests

        base_url = self.config["api.base_url"]
        response = requests.post(
            base_url + "/services/cluster/centroids/delete",
            headers={"Authorization": self.project + ":" + self.api_key},
            params={
                "dataset_id": dataset_id,
                "vector_field": vector_fields,
                "alias": self.alias,
            },
        )
        return response.json()["status"]

    def fit_predict(
        self,
        vector_fields: List[str],
        filters: Optional[List[Dict]] = None,
        return_only_clusters: bool = True,
        include_report: bool = True,
        update: bool = True,
        inplace: bool = False,
    ):
        """
        Parameters
        ----------
        data: Union[str, Dataset, List[Dict]]
            Either a reference to a Relevance AI Dataset, be it its name
            (string) or the object itself (Dataset), or a list of documents
            (List[Dict]).

        vector_fields: List[str]
            The vector fields over which to fit the model.

        filters: List[Dict]
            A list of filters to enable for document retrieval. This only
            applies to a reference to a Relevance AI Dataset.

        return_only_clusters: bool
            An indicator that determines what is returned. If True, this
            function returns the clusters. Else, the function returns the
            original documents.

        include_report: bool
            An indictor that determines whether to include (True) a report and grade
            base on the mean silhouette score or not (False).

        update: bool
            An indicator that determines whether to update the documents
            that were part of the clustering process. This only applies to a
            reference to a Relevance AI Dataset.

        inplace: bool
            An indicator that determines whether the documents are edited
            inplace (True) or a copy is created and edited (False).

        Example
        -------

        .. code-block::

            from relevanceai import ClusterBase, Client

            client = Client()

            import random
            class CustomClusterModel(ClusterBase):
                def fit_predict(self, X):
                    cluster_labels = [random.randint(0, 100) for _ in range(len(X))]
                    return cluster_labels

            model = CustomClusterModel()

            df = client.Dataset("sample_dataset")
            clusterer = client.ClusterOps(alias="random_clustering", model=model)
            clusterer.fit_predict_update(df, vector_fields=["sample_vector_"])

        """
        filters = [] if filters is None else filters

        self.vector_fields = vector_fields
        # make sure to only get fields where vector fields exist
        filters.extend(
            [
                {
                    "field": f,
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": " ",
                    "strict": "must_or",
                }
                for f in vector_fields
            ]
        )
        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )
        print("Retrieving all documents")
        documents = self._get_all_documents(
            dataset_id=self.dataset_id, filters=filters, select_fields=vector_fields
        )
        if len(documents) == 0:
            raise NoDocumentsError()

        vectors = self._get_vectors_from_documents(vector_fields, documents)

        # Label the clusters
        print("Fitting and predicting on all relevant documents")
        cluster_labels = self._label_clusters(self.model.fit_predict(vectors))

        if include_report:
            try:
                self._calculate_silhouette_grade(vectors, cluster_labels)
                report = ClusterReport(
                    X=vectors,
                    cluster_labels=cluster_labels,
                    num_clusters=len(cluster_labels),
                    model=self.model,
                )
                response = self.store_cluster_report(
                    report_name=self._get_cluster_field_name(self.alias),
                    report=report.internal_report,
                )
            except Exception as e:
                print(e)
                pass

        clustered_documents = self.set_cluster_labels_across_documents(
            cluster_labels,
            documents,
            inplace=inplace,
            return_only_clusters=return_only_clusters,
        )

        if update:
            # Updating the db
            print("Updating the database...")
            results = self._update_documents(
                self.dataset_id, clustered_documents, chunksize=10000
            )
            self.logger.info(results)

            # Update the centroid collection
            self.model.vector_fields = vector_fields

        self._insert_centroid_documents()
        print(
            "Build your clustering app here: "
            + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
        )

        return clustered_documents

    @track
    def fit(
        self,
        dataset_id: str,
        vector_fields: List[str],
        filters: Optional[List] = None,
        include_report: bool = True,
        verbose: bool = True,
    ):
        """
        This function fits a cluster model onto a dataset. It sits under `fit`
        and is usually the only function that runs unless you have a get_centroid_documents
        function.

        Parameters
        ---------------
        dataset: Union[Dataset, str],
            The dataset object to fit it on
        vector_fields: list
            The vector fields to fit the model on
        filters: list
            The filters to run it on
        include_filters_for_vector_fields: bool
            If True, only cluster on those with the vector fields inside it.

        Example
        ---------

        .. code-block::

            from relevanceai import ClusterBase, Client
            client = Client()

            import random
            class CustomClusterModel(ClusterBase):
                def fit_predict(self, X):
                    cluster_labels = [random.randint(0, 100) for _ in range(len(X))]
                    return cluster_labels

            model = CustomClusterModel()

            clusterer = client.ClusterOps(model, alias="random_clustering")
            df = client.Dataset("sample_dataset")

            clusterer.fit_predict_update(df, vector_fields=["sample_vector_"])

        """
        self.dataset_id = dataset_id
        self.vector_fields = vector_fields

        filters = [] if filters is None else filters

        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )

        # make sure to only get fields where vector fields exist
        filters += [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields
        ]
        if verbose:
            print("Retrieving all documents")
        fields_to_get = vector_fields.copy()
        if self.parent_alias:
            parent_field = self._get_cluster_field_name(self.parent_alias)
            fields_to_get.append(parent_field)

        # TODO: Figure out why this isn't cached
        docs = self._get_all_documents(
            dataset_id=dataset_id, filters=filters, select_fields=fields_to_get
        )

        if len(docs) == 0:
            raise NoDocumentsError()

        if verbose:
            print("Fitting and predicting on all documents")

        clustered_docs = self.fit_predict(
            vector_fields=vector_fields,
            return_only_clusters=True,
            inplace=False,
            include_report=include_report,
        )

        # Updating the db
        if verbose:
            print("Updating the database...")
        results = self._update_documents(dataset_id, clustered_docs, chunksize=10000)
        self.logger.info(results)

        # Update the centroid collection
        self.model.vector_fields = vector_fields

        self._insert_centroid_documents()

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{dataset_id}/deploy/recent/cluster"
            )

        return self

    def unique_cluster_ids(
        self,
        alias: str = None,
        minimum_cluster_size: int = 3,
        dataset_id: str = None,
        num_clusters: int = 1000,
    ):
        """
        We call facets on our data, which looks a little like this:

        {'results': {'_cluster_.sample_1_vector_.kmeans-3': [{'_cluster_.sample_1_vector_.kmeans-3': 'cluster-2',
                'frequency': 41,
                'value': 'cluster-2'},
            {'_cluster_.sample_1_vector_.kmeans-3': 'cluster-0',
                'frequency': 34,
                'value': 'cluster-0'},
            {'_cluster_.sample_1_vector_.kmeans-3': 'cluster-1',
                'frequency': 25,
                'value': 'cluster-1'}]}}
        """
        # Mainly to be used for subclustering
        # Get the cluster alias
        if dataset_id is None:
            self._check_for_dataset_id()
            dataset_id = self.dataset_id

        cluster_field = self._get_cluster_field_name(alias=alias)

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=dataset_id,
                fields=[cluster_field],
                page_size=int(self.config["data.max_clusters"]),
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if cluster_field not in facet_results:
                raise ValueError(
                    f"No clusters with alias `{alias}`. Please check the schema."
                )
            for facet in facet_results[cluster_field]:
                if facet["frequency"] > minimum_cluster_size:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[cluster_field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)
        return list(all_cluster_ids)

    @track
    def fit_dataset(
        self,
        dataset,
        vector_fields,
        filters: Optional[list] = None,
        return_only_clusters: bool = True,
        inplace=False,
        verbose: bool = True,
    ):
        """ """
        filters = [] if filters is None else filters

        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )

        self._init_dataset(dataset)
        self.vector_fields = vector_fields

        # make sure to only get fields where vector fields exist
        filters += [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields
        ]
        if verbose:
            print("Retrieving all documents")
        docs = self._get_all_documents(
            dataset_id=self.dataset_id, filters=filters, select_fields=vector_fields
        )

        if verbose:
            print("Fitting and predicting on all documents")
        return self.fit_predict(
            vector_fields,
            docs,
            return_only_clusters=True,
            inplace=inplace,
        )

    @track
    def predict_documents(
        self,
        vector_fields,
        documents,
        return_only_clusters: bool = True,
        inplace: bool = True,
    ):
        """
        Predict the documents
        """
        if len(vector_fields) > 1:
            raise ValueError("Currently do not suport more than 1 vector field.")
        vectors = self._get_vectors_from_documents(vector_fields, documents)
        cluster_labels = self.model.predict(vectors)
        cluster_labels = self._label_clusters(cluster_labels)
        return self.set_cluster_labels_across_documents(
            cluster_labels,
            documents,
            inplace=inplace,
            return_only_clusters=return_only_clusters,
        )

    @track
    def predict_update(
        self,
        dataset,
        vector_fields: Optional[List[str]] = None,
        chunksize: int = 20,
        verbose: bool = True,
    ):
        """
        Predict the dataset.

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="minibatchkmeans_2", model=model)

            cluster_ops.partial_fit_dataset(df)
            cluster_ops.predict_dataset(df)

        """
        if not vector_fields:
            vector_fields = self.vector_fields

        all_responses = {
            "inserted": 0,
            "failed_documents": [],
            "failed_documents_detailed": [],
        }

        filters = [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields
        ]

        for c in self._chunk_dataset(
            dataset, self.vector_fields, chunksize=chunksize, filters=filters
        ):
            cluster_predictions = self.predict_documents(
                vector_fields=vector_fields, documents=c
            )
            response = self.dataset._update_documents(
                dataset_id=self.dataset_id, documents=cluster_predictions
            )
            for k, v in response.items():
                if isinstance(all_responses[k], int):
                    all_responses["inserted"] += v
                elif isinstance(all_responses[k], list):
                    all_responses[k] += v
        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
        return all_responses

    def set_cluster_labels_across_documents(
        self,
        cluster_labels: list,
        documents: List[Dict],
        inplace: bool = True,
        return_only_clusters: bool = True,
    ):
        """
        Utility function to allow users to set cluster labels

        Parameters
        ------------
        cluster_labels: List[str, int]
            A list of integers of string. If it is an integer - it will automatically add a 'cluster-' prefix
            to help avoid incorrect data type parsing. You can override this behavior by setting clusters
            as strings.
        documents: List[dict]
            When the documents are in
        inplace: bool
            If True, then the clusters are set in place.
        return_only_clusters: bool
            If True, then the return_only_clusters will return documents with just the cluster field and ID.
            This can be helpful when you want to upsert quickly without having to re-insert the entire document.

        Example
        -----------

        .. code-block::

            labels = list(range(10))
            documents = [{"_id": str(x)} for x in range(10)]
            cluster_ops.set_cluster_labels_across_documents(labels, documents)

        """
        if inplace:
            self._set_cluster_labels_across_documents(cluster_labels, documents)
            if return_only_clusters:
                return [
                    {"_id": d.get("_id"), self.cluster_field: d.get(self.cluster_field)}
                    for d in documents
                ]
            return documents

        # useful if you want to upload as quickly as possible
        new_documents = documents.copy()

        self._set_cluster_labels_across_documents(cluster_labels, new_documents)
        if return_only_clusters:
            return [
                {"_id": d.get("_id"), self.cluster_field: d.get(self.cluster_field)}
                for d in new_documents
            ]
        return new_documents

    @property
    def metadata(self):
        """
        If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
        Otherwise, you can store the metadata about your cluster here.

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("_github_repo_vectorai")
            from relevanceai.ops.clusterops.kmeans_clusterer import KMeansModel

            model = KMeansModel()
            kmeans = client.ClusterOps(model, alias="kmeans_sample")
            kmeans.fit(df, vector_fields=["sample_1_vector_"])
            kmeans.metadata
            # {"k": 10}

        """
        return self.services.cluster.centroids.metadata(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )

    @metadata.setter
    def metadata(self, metadata: dict):
        """
        If metadata is none, retrieves metadata about a dataset. notably description, data source, etc
        Otherwise, you can store the metadata about your cluster here.

        Parameters
        ----------
        metadata: dict
           If None, it will retrieve the metadata, otherwise
           it will overwrite the metadata of the cluster

        Example
        ----------

        .. code-block::


            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)

            cluster_ops.fit(df, vector_fields=["sample_1_vector_"])
            cluster_ops.metadata
            # {"k": 10}

        """
        return self.services.cluster.centroids.metadata(
            dataset_id=self.dataset_id,
            vector_fields=self.vector_fields,
            alias=self.alias,
            metadata=metadata,
        )

    @track
    def evaluate(
        self,
        ground_truth_column: Union[str, None] = None,
        metric: Union[str, Callable] = "euclidean",
        random_state: Union[int, None] = None,
        average_score: bool = False,
    ):
        """
        Evaluate your clusters using the silhouette score, or if true labels are provided, using completeness, randomness and homogeniety as well

        Parameters
        ----------
        ground_truth_index : str
            the index of the true label of each sample in dataset

        metric : str or Callable
            A string referencing suportted distance functions, or custom method for calculating the distance between 2 vectors

        random_state : int, default None
            for reproducability

        average_score : bool
            a boolean that determines whether to average the evaluation metrics in a new a score. only applicable if ground_truth_index is not None

        Example
        ----------

        .. code-block::


            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset")

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=2)
            cluster_ops = client.ClusterOps(alias="kmeans_2", model=model)

            cluster_ops.fit(df, vector_fields=["sample_vector_"])

            cluster_ops.evaluate()
            cluster_ops.evaluate("truth_column")
        """

        from sklearn.metrics import (
            adjusted_rand_score,
            completeness_score,
            silhouette_score,
            homogeneity_score,
        )

        vector_field = self.vector_fields[0]
        cluster_field = self.cluster_field
        alias = self.alias

        samples = self._get_all_documents(self.dataset_id, include_vector=True)

        vectors = [sample[vector_field] for sample in samples]
        pred_labels = [sample[cluster_field][vector_field][alias] for sample in samples]

        def norm(value, min, max):
            return (value - min) / (max - min)

        s_score = silhouette_score(
            vectors, pred_labels, metric=metric, random_state=random_state
        )

        stats = {}
        stats["silhouette"] = {
            "score": s_score,
            "description": METRIC_DESCRIPTION["silhouette"],
        }

        if (
            ground_truth_column
        ):  # if the ground truth column was provided, we can run these cluster evaluation measures

            ground_truth_labels = {  # convert cluster1 etc. to respective label found in ground truth column
                key: value[0][ground_truth_column]
                for key, value in self.list_closest_to_center(page_size=1).items()
                if value
            }
            pred_labels = (
                [  # get the predicted clusters and return their respective true labels
                    ground_truth_labels[sample[cluster_field][vector_field][alias]]
                    for sample in samples
                ]
            )
            true_labels = [
                sample[ground_truth_column] for sample in samples
            ]  # same for actual labels

            ar_score = adjusted_rand_score(true_labels, pred_labels)  # compute metric
            c_score = completeness_score(true_labels, pred_labels)  # compute metric
            h_score = homogeneity_score(true_labels, pred_labels)  # compute metric

            stats["random"] = {
                "score": ar_score,
                "description": METRIC_DESCRIPTION["random"],
            }
            stats["completeness"] = {
                "score": c_score,
                "description": METRIC_DESCRIPTION["completeness"],
            }
            stats["homogeneity"] = {
                "score": h_score,
                "description": METRIC_DESCRIPTION["homogeneity"],
            }

            if average_score:  # If the user wants an average as well
                ns_score = norm(s_score, min=-1, max=1)
                nar_score = norm(ar_score, min=-1, max=1)

                average_score = (ns_score + nar_score + c_score + h_score) / 4

                stats["average"] = {
                    "score": average_score,
                    "description": METRIC_DESCRIPTION["average"],
                }

        return stats

    @beta
    @track
    def store_cluster_report(
        self, report_name: str, report: dict, verbose: bool = True
    ):
        """

        Store the cluster data.

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.store_cluster_report("sample", {"value": 3})

        """
        response: dict = self.reports.clusters.create(
            name=report_name, report=self.json_encoder(report)
        )
        if verbose:
            print(
                f"You can now access your report at https://cloud.relevance.ai/report/cluster/{self.region}/{response['_id']}"
            )
        return response

    @track
    def internal_report(self, verbose: bool = True):
        """
        Get a report on your clusters.

        Example
        ---------

        This is what is returned on an `auto_cluster` method.

        .. code-block::

            from relevanceai.package_utils.datasets import mock_documents
            docs = mock_documents(10)
            df = client.Dataset('sample')
            df.upsert_documents(docs)
            cluster_ops = df.auto_cluster('kmeans-2', ['sample_1_vector_'])
            cluster_ops.internal_report()

        This is what is returned on an `internal_report` method.

        .. code-block::

            from relevanceai import Client
            # client = Client()
            from relevanceai.package_utils.datasets import mock_documents
            ds = client.Dataset("sample")
            # Creates 100 sample documents
            documents = mock_documents(100)
            ds.upsert_documents(documents)

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=10)
            clusterer = client.ClusterOps(alias="not-auto-kmeans-10", model=model)

            clusterer.fit_predict_update(dataset=ds, vector_fields=["sample_1_vector_"])
            cluster_ops.internal_report()

        """
        if isinstance(self.vector_fields, list) and len(self.vector_fields) > 1:
            raise ValueError(
                "We currently do not support more than 1 vector field when reporting."
            )
        from relevanceai.reports.cluster import ClusterReport

        # X is all the vectors
        cluster_field_name = self._get_cluster_field_name()
        all_docs = self._get_all_documents(
            self.dataset_id, select_fields=self.vector_fields + [cluster_field_name]
        )
        cluster_labels = self.get_field_across_documents(cluster_field_name, all_docs)
        self.number_of_clusters = len(set(cluster_labels))
        self._report = ClusterReport(
            self.get_field_across_documents(self.vector_fields[0], all_docs),
            cluster_labels=self.get_field_across_documents(
                cluster_field_name, all_docs
            ),
            model=self.model,
            num_clusters=self.number_of_clusters,
        )
        cluster_response = self.store_cluster_report(
            report_name=cluster_field_name,
            report=self.json_encoder(self._report.internal_report),
            verbose=verbose,
        )

        return self._report.internal_report

    report = internal_report

    def operate(self, field: str, func: Callable, output_field: Optional[str] = None):
        """
        Run an function per cluster.

        Parameters
        ------------

        field: str
            The field to operate on
        func: Callable
            The function to run on all the values once they are received.
            It should take in a list of values
        output_field: Optional[str]
            Outputs for every document in the cluster

        Example
        ---------

        .. code-block::

            import numpy as np
            def vector_mean(vectors):
                return np.mean(vectors, axis=0)
            cluster_centroids = cluster_ops.operate(
                field="review_sentence_answer_use_vector_",
                func=vector_mean
            )

        """
        # Run a function on each cluster
        output: Dict = {}
        cluster_ids = self.unique_cluster_ids()
        for cluster_id in tqdm(cluster_ids):
            self._operate(cluster_id, field, output, func)
        if output_field is not None:
            self.update_documents_within_clusters(output, output_field)
        return output

    def create_centroids(
        self, vector_fields: List[str], operation: Optional[Callable] = None
    ):
        """
        Create centroids if there are none. The default operation is to take the centroid
        of each vector.
        An alternative function can be provided provided.

        Example of the operation in question for mean:

        .. code-block::

            def vector_mean(vectors):
                return np.mean(vectors, axis=0)

            clusterops.create_centroids(["sample_vector_"], operation=vector_mean)

        """
        if len(vector_fields) > 1:
            raise ValueError("currently do not support more than 1 vector field.")
        vector_field: str = vector_fields[0]

        def vector_mean(vectors):
            return np.mean(vectors, axis=0)

        if operation == "mean" or operation is None:
            operation = vector_mean

        cluster_centroids = self.operate(field=vector_field, func=operation)  # type: ignore
        centroid_docs = []
        for k, v in cluster_centroids.items():
            centroid_docs.append({"_id": str(k), vector_field: v.tolist()})
        return self.insert_centroid_documents(centroid_docs)

    def update_documents_within_cluster(self, cluster_id: str, update: dict):
        """
        Update all the documents within a cluster
        """
        cluster_filter = self._get_filter_for_cluster(cluster_id)
        result = self.datasets.documents.update_where(
            dataset_id=self.dataset_id, update=update, filters=cluster_filter
        )
        return result

    def update_documents_within_clusters(
        self, cluster_id_and_updates: dict, output_field: str = None
    ):
        """
        Takes the cluster ids and updates and updates them accordingly

        Example
        ---------

        .. code-block::

            # Let us take the operator results
            operator_results = clusterops.operate(...)
            clusterops.update_documents_within_clusters(
                operator_results, output_field="centroid_vector_"
            )
        """
        results = []
        for cluster_id, update_value in cluster_id_and_updates.items():
            if type(update_value) != dict:
                if output_field is None:
                    raise ValueError(
                        """
                        As update value is not a dictionary, you will need to
                        specify `output_field=`.
                    """
                    )
                update: dict = {}
                self.set_field(output_field, update, update_value)
            else:
                update = update_value

            cluster_filter = self._get_filter_for_cluster(cluster_id)
            result = self.datasets.documents.update_where(
                dataset_id=self.dataset_id, update=update, filters=cluster_filter
            )
            results.append(result)
        return results

    # def show(
    #     self,
    #     dataset_id: str,
    #     field: str,
    #     vector_field: str,
    #     alias: str,
    #     preview_num: int = 10,
    #     is_image_field: bool = False,
    # ):
    #     """
    #     Shows the values of a clustering.

    #     dataset_id: str
    #         The dataset ID of with the clustering of interest.

    #     field: str
    #         The field whose values are of interest.

    #     vector_field: str
    #         The vector field that was used for the clustering.

    #     alias: str
    #         The alias of the clustering.

    #     preview_num: int
    #         The maximum number of values to preview. If a cluster has fewer
    #         values than preview_num, the cluster list will be extended by
    #         the appropriate number of np.nans to make up the difference.

    #     is_image_field: bool
    #         If set to True, will show the values as the images rather than
    #         links.
    #     """
    #     if type(preview_num) is not int and preview_num <= 0:
    #         raise TypeError(
    #             f"Please provide a valid non-zero integer for {preview_num}."
    #         )

    #     if not vector_field.startswith(field):
    #         raise ValueError(f"{vector_field} must be a vector of {field}.")

    #     # Since there is no gaurantee self.dataset_id will be not None, it is
    #     # safer to force the user to specify the dataset_id
    #     ds = Dataset(
    #         self.project, self.api_key, dataset_id, self.firebase_uid, fields=[]
    #     )
    #     schema = ds.schema

    #     if field not in schema:
    #         raise ValueError(f"{field} does not exist")

    #     cluster_field = ".".join(["_cluster_", vector_field])
    #     if cluster_field not in schema:
    #         raise ValueError(f"{vector_field} has not been clustered yet.")

    #     alias_field = ".".join([cluster_field, alias])
    #     if alias_field not in schema:
    #         raise ValueError(f"A clustering with alias {alias} has not been made yet.")

    #     cluster_values = ds.to_pandas_dataframe(
    #         select_fields=[alias_field, field], show_progress_bar=False
    #     )
    #     # converts {"vector_field": {"alias": "cluster-k"}} to "cluster-k"
    #     cluster_values["_cluster_"] = cluster_values["_cluster_"].apply(
    #         lambda cluster: cluster[vector_field][alias]
    #     )

    #     cluster_groups = dict(list(cluster_values.groupby("_cluster_")[field]))
    #     clusters = {}
    #     for cluster, values in cluster_groups.items():
    #         length = min(len(values), preview_num)
    #         # It is faster to index a list than a pandas Series
    #         clusters[cluster] = values.tolist()[:length]

    #         # If the number of values in the cluster is fewer than the desired
    #         # preview number, insert np.nan to make up the difference. Note
    #         # that the range of a negative number is not a range.
    #         difference = preview_num - len(clusters[cluster])
    #         clusters[cluster].extend([np.nan for _ in range(difference)])

    #     # return pd.DataFrame(clusters)
    #     return _ClusterOpsShow(pd.DataFrame(clusters), is_image_field)
