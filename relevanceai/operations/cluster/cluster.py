from typing import Any, Set, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from doc_utils import DocUtils
from relevanceai.client.helpers import Credentials
from relevanceai.operations import BaseOps
from relevanceai.utils.decorators import beta, track, deprecated
from relevanceai.constants import (
    Warning,
    Messages,
    ModelNotSupportedError,
    CLUSTER_APP_LINK,
)
from relevanceai.operations.cluster.models.summarizer import TransformersLMSummarizer

from relevanceai.operations.cluster.utils import ClusterUtils

from relevanceai.utils import DocUtils
from relevanceai.utils.distances import (
    euclidean_distance_matrix,
    cosine_similarity_matrix,
)


class ClusterWriteOps(ClusterUtils, BaseOps, DocUtils):
    """
    You can load ClusterOps instances in 2 ways.

    .. code-block::

        # State the vector fields and alias in the ClusterOps object
        cluster_ops = client.ClusterOps(
           alias="kmeans-25",
            dataset_id="sample_dataset_id",
            vector_fields=["sample_vector_"]
        )
        cluster_ops.list_closest()

        # State the vector fields and alias in the operational call
        cluster_ops = client.ClusterOps(alias="kmeans-25")
        cluster_ops.list_closest(
            dataset="sample_dataset_id",
            vector_fields=["sample_vector_"]
        )

    """

    dataset_id: str
    cluster_field: str

    def __init__(
        self,
        credentials: Credentials,
        model: Any = None,
        alias: str = None,
        n_clusters: Optional[int] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        outlier_value: int = -1,
        outlier_label: str = "outlier",
        verbose: bool = True,
        vector_fields: Optional[list] = None,
        **kwargs,
    ):
        """
        ClusterOps object

        Parameters
        -------------

        model: Any
            The string of clustering algorithm, class of clustering algorithm or custom clustering class.
            If custom, the model must contain the method for fit_predict and must output a numpy array for labels

        alias: str = None
            An alias to be given for clustering.
            If no alias is provided, alias becomes the cluster model name hypen n_clusters e.g. (kmeans-5, birch-8)

        n_clusters: Optional[int] = None
            Number of clusters to find, should the argument be required of the algorithm.
            If None, n_clusters defaults to 8 for sklearn models ONLY.

        cluster_config: Optional[Dict[str, Any]] = None
            Hyperparameters for the clustering model.
            See the documentation for supported clustering models for list of hyperparameters

        outlier_value: int = -1
            For unsupervised clustering models like OPTICS etc. outliers are given the integer label -1

        outlier_label: str = "outlier"
            When viewing the cluster app dashboard, outliers will be prefixed with outlier_label

        """

        self.cluster_config = (
            {} if cluster_config is None else cluster_config
        )  # type: ignore

        self.model_name = None
        self.verbose = verbose

        if model is None:
            model = "kmeans"
            if verbose:
                print(f"No clustering model selected: defaulting to `{model}`")

        if isinstance(model, str):
            supervised = model.lower() not in [
                "hdscan",
                "optics",
                "dbscan",
                "communitydetection",
            ]
        else:
            supervised = False

        self.n_clusters = n_clusters
        if "n_clusters" in self.cluster_config and supervised:
            self.n_clusters = self.cluster_config["n_clusters"]
            n_clusters = self.cluster_config["n_clusters"]

        if supervised:
            if n_clusters is not None:
                self.cluster_config["n_clusters"] = n_clusters  # type: ignore
            else:
                self.cluster_config["n_clusters"] = 25  # type: ignore

        self.model = self._get_model(model)

        if self.n_clusters is None:
            if hasattr(self.model, "n_clusters"):
                self.n_clusters = self.n_clusters

            elif hasattr(self.model, "k"):
                self.n_clusters = self.model.k

        if "package" not in self.__dict__:
            self.package = self._get_package(self.model)

        self.alias = self._get_alias(alias)
        self.vector_fields = vector_fields  # type: ignore
        if self.vector_fields is not None and len(self.vector_fields) >= 1:
            self.vector_field = self.vector_fields[0]
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label

        for key, value in kwargs.items():
            if not hasattr(self, key):
                if key == "vector_fields":
                    setattr(self, "vector_field", value[0])
                setattr(self, key, value)

        super().__init__(credentials)

    def __call__(
        self,
        dataset_id: str,
        vector_fields: Optional[List[str]] = None,
        include_cluster_report: bool = True,
        **kwargs,
    ) -> None:
        return self.run(
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            include_cluster_report=include_cluster_report,
            **kwargs,
        )

    @deprecated(version="1.0.0")
    def fit_predict_update(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def _get_schema(self) -> Dict:
        return self.datasets.schema(dataset_id=self.dataset_id)

    def _get_alias(self, alias: Any) -> str:
        # Auto-generates alias here
        if alias is None:
            if hasattr(self.model, "n_clusters"):
                n_clusters = (
                    self.n_clusters
                    if self.n_clusters is not None
                    else self.model.n_clusters
                )
                alias = f"{self.model_name}-{n_clusters}"

            elif hasattr(self.model, "k"):
                n_clusters = (
                    self.n_clusters if self.n_clusters is not None else self.model.k
                )
                alias = f"{self.model_name}-{n_clusters}"

            else:
                alias = self.model_name

            Warning.MISSING_ALIAS.format(alias=alias)

        if self.verbose:
            print(f"The alias is `{alias.lower()}`.")
        return alias.lower()

    def _get_package(self, model):
        model_name = str(model.__class__).lower()
        if "function" in model_name:
            model_name = str(model.__name__)

        if "sklearn" in model_name:
            package = "sklearn"

        elif "faiss" in model_name:
            package = "faiss"

        elif "hdbscan" in model_name:
            package = "hdbscan"

        elif "communitydetection" in model_name:
            package = "sentence-transformers"

        else:
            package = "custom"

        return package

    def _get_model(self, model):
        if isinstance(model, str):
            model = model.lower().replace(" ", "").replace("_", "")
            self.model_name = model
        else:
            self.model_name = str(model.__class__).split(".")[-1].split("'>")[0]
            self.package = "custom"
            return model

        if isinstance(model, str):
            if model == "affinitypropagation":
                from sklearn.cluster import AffinityPropagation

                model = AffinityPropagation(**self.cluster_config)

            elif model == "agglomerativeclustering":
                from sklearn.cluster import AgglomerativeClustering

                model = AgglomerativeClustering(**self.cluster_config)

            elif model == "birch":
                from sklearn.cluster import Birch

                model = Birch(**self.cluster_config)

            elif model == "dbscan":
                from sklearn.cluster import DBSCAN

                model = DBSCAN(**self.cluster_config)

            elif model == "optics":
                from sklearn.cluster import OPTICS

                model = OPTICS(**self.cluster_config)

            elif model == "kmeans":
                from sklearn.cluster import KMeans

                if (
                    self.n_clusters is None
                    and self.cluster_config.get("n_clusters", None) is None
                ):
                    self.cluster_config["n_clusters"] = 25
                model = KMeans(**self.cluster_config)

            elif model == "featureagglomeration":
                from sklearn.cluster import FeatureAgglomeration

                model = FeatureAgglomeration(**self.cluster_config)

            elif model == "meanshift":
                from sklearn.cluster import MeanShift

                model = MeanShift(**self.cluster_config)

            elif model == "minibatchkmeans":
                from sklearn.cluster import MiniBatchKMeans

                model = MiniBatchKMeans(**self.cluster_config)

            elif model == "spectralclustering":
                from sklearn.cluster import SpectralClustering

                model = SpectralClustering(**self.cluster_config)

            elif model == "spectralbiclustering":
                from sklearn.cluster import SpectralBiclustering

                model = SpectralBiclustering(**self.cluster_config)

            elif model == "spectralcoclustering":
                from sklearn.cluster import SpectralCoclustering

                model = SpectralCoclustering(**self.cluster_config)

            elif model == "hdbscan":
                from hdbscan import HDBSCAN

                model = HDBSCAN(**self.cluster_config)

            elif model in "communitydetection":
                from relevanceai.operations.cluster.algorithms import CommunityDetection

                model = CommunityDetection(**self.cluster_config)

            elif "faiss" in model:
                from faiss import Kmeans

                model = Kmeans(**self.cluster_config)
            else:
                raise ValueError(
                    """Invalid model. This should be one of ['affinitypropagation',
                    'agglomerativeclustering', 'birch', dbscan', 'optics', 'kmeans',
                    'featureagglomeration', 'meanshift', 'minibatchkmeans',
                    'spectralclustering', 'spectralbiclustering', 'spectralcoclustering',
                    'hdbscan', 'communitydetection']
                    ]"""
                )

        else:
            raise ModelNotSupportedError

        return model

    def _format_labels(self, labels: np.ndarray) -> List[str]:
        labels = labels.flatten().tolist()
        cluster_labels = [
            f"cluster-{str(label)}"
            if label != self.outlier_value
            else self.outlier_label
            for label in labels
        ]
        return cluster_labels

    def _get_centroid_documents(
        self, vectors: np.ndarray, labels: List[str], vector_field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        centroid_documents = []

        centroids: Dict[str, Any] = {}
        for label, vector in zip(labels, vectors.tolist()):
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(vector)

        for centroid, vectors in centroids.items():
            centroid_vector = np.array(vectors).mean(0).tolist()
            if vector_field:
                centroid_document = {"_id": centroid, vector_field: centroid_vector}
            else:
                centroid_document = dict(_id=centroid, centroid_vector=centroid_vector)
            centroid_documents.append(centroid_document)

        return centroid_documents

    def _get_document_vector_field(self):
        vector_fields = []
        metadata = self.datasets.metadata(self.dataset_id)
        most_recent_updated_vector = max(
            metadata["results"]["_vector_"].items(), key=lambda x: x[1]
        )[0]
        vector_fields.append(most_recent_updated_vector)
        return vector_fields

    def _retrieve_dataset_id(self, dataset) -> str:
        """Helper method to get multiple dataset values.
        Dataset can be either a Dataset or a string object."""
        from relevanceai.dataset import Dataset

        if isinstance(dataset, Dataset):
            dataset_id: str = dataset.dataset_id
        elif isinstance(dataset, str):
            dataset_id = dataset
        elif dataset is None:
            if hasattr(self, "dataset_id"):
                # let's not surprise users
                print(
                    f"No dataset supplied - using last stored one '{self.dataset_id}'."
                )
                dataset_id = str(self.dataset_id)
            else:
                raise ValueError("Please supply dataset.")
        return dataset_id

    def _insert_centroids(
        self,
        dataset_id: str,
        vector_fields: List[str],
        centroid_documents: List[Dict[str, Any]],
    ) -> None:
        self.datasets.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=vector_fields,
            alias=self.alias,
        )

    def _insert_metadata(
        self, dataset_id: str, vector_fields: List[str], centroid_documents: List[Dict]
    ):
        metadata = self.datasets.metadata(dataset_id=dataset_id)
        # store in metadata
        if "_cluster_" not in metadata:
            metadata["_cluster_"] = {}

        # calculate dist matrix
        vectors = [
            centroid_document[vector_fields[0]]
            for centroid_document in centroid_documents
        ]
        metadata["_cluster_"][self.cluster_field] = {
            "vector_fields": vector_fields,
            "alias": self.alias,
            "params": {},  # TBC
            "similarity_matrix": {
                "euclidean": euclidean_distance_matrix(vectors, vectors, decimal=3),
                "cosine": cosine_similarity_matrix(vectors, vectors, decimal=3),
            },
        }

        self.datasets.post_metadata(dataset_id, metadata)

    def _fit_predict(
        self, documents: List[Dict[str, Any]], vector_field: str, inplace=True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        vectors = np.array(
            [self.get_field(vector_field, document) for document in documents]
        )

        if self.package == "sklearn":
            labels = self.model.fit_predict(vectors)

        elif self.package == "hdbscan":
            self.model.fit(vectors)
            labels = self.model.labels_.reshape(-1, 1)

        elif self.package == "faiss":
            vectors = vectors.astype("float32")
            self.model.train(vectors)
            labels = self.model.assign(vectors)[1]

        elif self.package == "sentence-transformers":
            # TODO: make teh config here better
            labels = self.model(vectors)
            labels = np.array(labels)

        elif self.package == "custom":
            labels = self.model.fit_predict(vectors)
            if not isinstance(labels, np.ndarray):
                raise ValueError("Custom model must output np.ndarray for labels")

            if labels.shape[0] != vectors.shape[0]:
                raise ValueError("incorrect shape")

        labels = self._format_labels(labels)

        if self.n_clusters is None:
            n_clusters = len(set(labels)) - 1
            print(f"Found {n_clusters} clusters using {self.model_name}")

        labelled_documents = [{"_id": d["_id"]} for d in documents]

        self.set_field_across_documents(
            field=self.cluster_field, values=labels, docs=labelled_documents
        )

        if inplace:  # add the cluster labels into the original documents
            self.set_field_across_documents(
                field=self.cluster_field, values=labels, docs=documents
            )

        centroid_documents = self._get_centroid_documents(
            vectors, labels, vector_field=vector_field
        )

        return centroid_documents, labelled_documents

    def _print_app_link(self):
        link = CLUSTER_APP_LINK.format(self.dataset_id)
        print(Messages.BUILD_HERE + link)

    @track
    def run(
        self,
        dataset_id: str,
        vector_fields: Optional[List[str]] = None,
        filters: Optional[list] = None,
        show_progress_bar: bool = True,
        verbose: bool = True,
        include_cluster_report: bool = True,
        report_name: str = "cluster-report",
    ) -> None:
        """
        Run clustering on a dataset

        Parameters
        --------------

        dataset_id: Optional[Union[str, Any]]
            The dataset ID
        vector_fields: Optional[List[str]]
            List of vector fields
        show_progress_bar: bool
            If True, the progress bar can be shown

        """
        filters = [] if filters is None else filters
        if not isinstance(dataset_id, str):
            if hasattr(dataset_id, "dataset_id"):
                dataset_id = dataset_id.dataset_id  # type: ignore
        self.dataset_id = dataset_id

        if vector_fields is None:
            vector_fields = self._get_document_vector_field()
            print(f"No vector_field given: defaulting to {vector_fields}")

        vector_field = vector_fields[0]
        self.cluster_field = f"_cluster_.{vector_fields[0]}.{self.alias}"

        # get all documents
        print("Retrieving all documents...")
        from relevanceai.utils.filter_helper import create_filter

        filters += create_filter(vector_field, filter_type="exists")
        documents = self._get_all_documents(
            dataset_id=dataset_id,
            select_fields=vector_fields,
            show_progress_bar=show_progress_bar,
            include_vector=True,
            filters=filters,
        )

        # fit model, predict and label all documents
        print("Predicting on all documents...")
        centroid_documents, labelled_documents = self._fit_predict(
            documents=documents, vector_field=vector_field, inplace=True
        )

        print("Updating cluster labels...")
        results = self._update_documents(
            dataset_id=dataset_id,
            documents=labelled_documents,
            show_progress_bar=show_progress_bar,
        )

        print("Inserting Centroids...")
        self._insert_centroids(
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            centroid_documents=centroid_documents,
        )
        print("Inserting Metadata...")
        self._insert_metadata(
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            centroid_documents=centroid_documents,
        )

        if include_cluster_report:
            # this needs to be more optimized for performance, we dont need to store 2 vector sets in memory.
            print("Generating evaluation report for your clusters…")
            from relevanceai.reports.cluster.report import ClusterReport

            centroids = self.get_field_across_documents(
                vector_field, centroid_documents, missing_treatment="raise_error"
            )
            centroids = [
                [round(value, 3) for value in centroid] for centroid in centroids
            ]

            X = self.get_field_across_documents(
                vector_field, documents, missing_treatment=self.outlier_value
            )

            cluster_labels = self.get_field_across_documents(
                self.cluster_field, documents, missing_treatment=-1
            )

            if len(cluster_labels) != len(X):
                raise ValueError(
                    "Number of cluster labels do not match number of rows of data."
                )

            self.report = ClusterReport(
                X=X,
                cluster_labels=cluster_labels,
                model=self.model,
                outlier_label=-1,
                centroids=centroids,
                verbose=True,
                include_typecheck=False,
            )

            try:
                response = self.reports.clusters.create(
                    name=report_name,
                    report=self.json_encoder(self.report.internal_report),
                )

                if verbose:
                    print(
                        f"📊 You can now access your report at https://cloud.relevance.ai/report/cluster/{self.region}/{response['_id']}"
                    )
            except Exception as e:
                print("Error creating cluster report! " + str(e))

        # link back to dashboard
        if verbose:
            self._print_app_link()

    def cluster_report(self, X, cluster_labels, centroids):
        from relevanceai.reports.cluster.report import ClusterReport

        cluster_labels = self.fit_predict(X)
        centroids = ClusterReport.calculate_centroids(X, cluster_labels)
        report = ...
        response = self.store_cluster_report(
            report_name="kmeans", report=report.internal_report
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
            )
        return self._centroids

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
        self._insert_centroids(
            dataset_id=self.dataset_id,
            vector_fields=[self.vector_fields[0]],
            centroid_documents=centroid_vectors,
        )
        return centroid_vectors

    def insert_centroids(self, centroid_documents):
        """
        Insert your own centroids

        Example
        ----------

        .. code-block::

            ds = client.Dataset("sample")
            cluster_ops = ds.ClusterOps(
                vector_fields=["sample_vector_"],
                alias="simple"
            )
            cluster_ops.insert_centroids(
                [
                    {
                        "_id": "cluster-1",
                        "sample_vector_": [1, 1, 1]
                    }
                ]
            )

        """
        results = self.datasets.cluster.centroids.insert(
            dataset_id=self.dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=self.vector_fields,
            alias=self.alias,
        )
        return results


class ClusterOps(ClusterWriteOps):
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
        dataset=None,
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
            dataset_id=self._retrieve_dataset_id(dataset),
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

    @track
    def merge(
        self,
        cluster_labels: List,
        alias: Optional[str] = None,
        show_progress_bar: bool = True,
        **update_kwargs,
    ):
        """
        Parameters
        ----------
        cluster_labels : Tuple[int]
            a tuple of integers representing the cluster ids you would like to merge
        alias: str
            the alias of the clustering you like to merge labels within
        show_progress_bar: bool
            whether or not to show the progress bar
        Example
        -------
        .. code-block::
            dataset.cluster(
                model="kmeans",
                n_clusters=3,
                vector_fields=["sample_1_vector_"],
            )
            ops = ClusterOps.from_dataset(
                dataset=dataset,
                alias="kmeans-3",
                vector_fields=["sample_1_vector_"],
            )
            ops.merge(cluster_labels=(0, 1), alias="kmeans-3")
        """

        if alias is None:
            if hasattr(self, "alias"):
                alias = self.alias
            else:
                raise ValueError("Please specify alias= as it was not detected")

        centroid_documents = self.datasets.cluster.centroids.documents(
            dataset_id=self.dataset_id,
            alias=self.alias,
            vector_fields=self.vector_fields,
            cluster_ids=cluster_labels,
            include_vector=True,
        )["results"]

        update: dict = {}

        if isinstance(cluster_labels[0], str):
            self.clusters = [cluster for cluster in sorted(cluster_labels)]
            self.min_cluster = cluster_labels[0]
        else:
            self.clusters = [f"cluster-{cluster}" for cluster in sorted(cluster_labels)]
            self.min_cluster = f"cluster-{min(cluster_labels)}"

        print(f"Merging clusters to {cluster_labels[0]}")
        update = {f"_cluster_.{self.vector_field}.{self.alias}": cluster_labels[0]}

        results = self.datasets.documents.update_where(
            dataset_id=self.dataset_id,
            update=update,
            filters=[
                {
                    "field": self._get_cluster_field_name(alias=alias),
                    "filter_type": "categories",
                    "condition": "==",
                    "condition_value": cluster_labels[1:],
                }
            ],
        )
        if results["status"] == "success":
            print("✅ Merged successfully.")
        else:
            print(f"🚨 Couldn't merge. : {results['message']}")

        try:
            # Calculating the centorids
            relevant_centroids = [
                self.get_field(self.vector_fields[0], d) for d in centroid_documents
            ]

            if len(relevant_centroids) == 0:
                raise ValueError("No relevant centroids found.")
            new_centroid = np.array(relevant_centroids).mean(0).tolist()

            if isinstance(cluster_labels[0], int):
                new_centroid_doc = {
                    "_id": f"cluster-{cluster_labels[0]}",
                    self.vector_field: new_centroid,
                }
            elif isinstance(cluster_labels[0], str):
                new_centroid_doc = {
                    "_id": cluster_labels[0],
                    self.vector_field: new_centroid,
                }

            # If there are no centroids - move on
            self.datasets.cluster.centroids.update(
                dataset_id=self.dataset_id,
                vector_fields=[self.vector_field],
                alias=alias,
                cluster_centers=[new_centroid_doc],
            )

            cluster: int

            for cluster in cluster_labels[1:]:
                if isinstance(cluster, str):
                    centroid_id = cluster
                else:
                    centroid_id = f"cluster-{cluster}"
                self.datasets.cluster.centroids.delete(
                    dataset_id=self.dataset_id,
                    centroid_id=centroid_id,
                    alias=self.alias,
                    vector_fields=[self.vector_field],
                )

            print("✅ Updated centroids.")

        except Exception as e:
            import traceback

            traceback.print_exc()
            pass

    @track
    def closest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
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
        verbose: bool = True,
    ):
        """
        List of documents closest from the center.

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        vector_field: list
            The vector field where a clustering task was run.
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
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        return self.datasets.cluster.centroids.list_closest_to_center(
            dataset_id=dataset_id,
            vector_fields=[vector_field],
            alias=alias,
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
            verbose=verbose,
        )

    @track
    def furthest(
        self,
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
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
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        return self.datasets.cluster.centroids.list_furthest_from_center(
            dataset_id=dataset_id,
            vector_fields=[vector_field],
            alias=alias,
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

    # Convenience functions
    list_closest = closest
    list_furthest = furthest

    # Summary functions
    @staticmethod
    def get_cluster_summary(
        summarizer,
        docs: Dict,
        summarize_fields: List[str],
        max_length: int = 100,
        first_sentence_only: bool = True,
    ):
        def _clean_sentence(s):
            s = (
                s.replace(". .", ".")
                .replace(" .", ".")
                .replace("\n", "")
                .replace("..", ".")
                .strip()
            )
            if s[-1] != ".":
                s += "."
            return s

        cluster_summary = {}
        for cluster, results in docs["results"].items():
            for f in summarize_fields:
                summary_fields = [
                    _clean_sentence(d[f])
                    for d in results["results"]
                    if d.get(f) and d[f] not in [" ", "."]
                ]
                summary_output = summarizer(
                    " ".join(summary_fields), max_length=max_length
                )[0]["summary_text"]
                summary = summary_output.replace(" .", ".").strip()
            if first_sentence_only:
                cluster_summary[cluster] = summary.split(".")[0]
            else:
                cluster_summary[cluster] = summary
        return cluster_summary

    @beta
    @track
    def summarize_closest(
        self,
        summarize_fields: List[str],
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
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
        model_name: str = "philschmid/bart-large-cnn-samsum",
        tokenizer: Optional[str] = None,
        max_length: int = 100,
        deployable_id: Optional[str] = None,
        first_sentence_only: bool = True,
        **kwargs,
    ):
        """
        List of documents closest from the center.

        Parameters
        ----------
        summarize_fields: list
            Fields to perform summarization, empty array/list means all fields
        dataset_id: string
            Unique name of dataset
        vector_field: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        centroid_vector_fields: list
            Vector fields stored
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
        model_name: str
            Huggingface Model to use for summarization.
            Pick from
            https://huggingface.co/models?pipeline_tag=summarization&sort=downloadshttps://huggingface.co/models?pipeline_tag=summarization
        tokenizer: str
            Tokenizer to use for summarization, allows you to bring your own tokenizer,
            else will instantiate pre-trained from selected model

        """
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        if not tokenizer:
            tokenizer = model_name

        if not hasattr(self, "summarizer"):
            self.summarizer = TransformersLMSummarizer(model_name, tokenizer, **kwargs)

        center_docs = self.list_closest(
            select_fields=summarize_fields,
            dataset_id=dataset_id,
            vector_field=vector_field,
            alias=alias,
            cluster_ids=cluster_ids,
            centroid_vector_fields=centroid_vector_fields,
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

        cluster_summary = self.get_cluster_summary(
            self.summarizer,
            docs=center_docs,
            summarize_fields=summarize_fields,
            max_length=max_length,
            first_sentence_only=first_sentence_only,
        )

        if deployable_id is not None:
            if dataset_id is None:
                if not hasattr(self, "dataset_id"):
                    raise ValueError("You need a dataset ID to update.")
                else:
                    dataset_id = self.dataset_id
            configuration = self.deployables.get(deployable_id=deployable_id)
            configuration["cluster-labels"] = cluster_summary
            self.deployables.update(
                deployable_id=deployable_id,
                dataset_id=dataset_id,
                configuration=configuration,
            )
        return {"results": cluster_summary}

    @beta
    @track
    def summarize_furthest(
        self,
        summarize_fields: List[str],
        dataset_id: Optional[str] = None,
        vector_field: Optional[str] = None,
        alias: Optional[str] = None,
        cluster_ids: Optional[List] = None,
        centroid_vector_fields: Optional[List] = None,
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
        model_name: str = "sshleifer/distilbart-cnn-6-6",
        tokenizer: Optional[str] = None,
        **kwargs,
    ):
        """
        List of documents furthest from the center.

        Parameters
        ----------
        summarize_fields: list
            Fields to perform summarization, empty array/list means all fields
        dataset_id: string
            Unique name of dataset
        vector_field: list
            The vector field where a clustering task was run.
        cluster_ids: list
            Any of the cluster ids
        alias: string
            Alias is used to name a cluster
        centroid_vector_fields: list
            Vector fields stored
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
        model_name: str
            Huggingface Model to use for summarization.
            Pick from
            https://huggingface.co/models?pipeline_tag=summarization&sort=downloadshttps://huggingface.co/models?pipeline_tag=summarization
        tokenizer: str
            Tokenizer to use for summarization, allows you to bring your own tokenizer,
            else will instantiate pre-trained from selected model

        """
        dataset_id = self.dataset_id if dataset_id is None else dataset_id
        vector_field = self.vector_field if vector_field is None else vector_field
        alias = self.alias if alias is None else alias

        if not tokenizer:
            tokenizer = model_name
        summarizer = TransformersLMSummarizer(model_name, tokenizer)

        center_docs = self.list_furthest(
            select_fields=summarize_fields,
            dataset_id=dataset_id,
            vector_field=vector_field,
            alias=alias,
            cluster_ids=cluster_ids,
            centroid_vector_fields=centroid_vector_fields,
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

        cluster_summary = self.get_cluster_summary(
            summarizer, docs=center_docs, summarize_fields=summarize_fields
        )

        return {"results": cluster_summary}
