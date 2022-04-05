import numpy as np
from typing import Optional, List, Any, Union, Dict, Tuple, Set
from tqdm.auto import tqdm
from relevanceai.operations.cluster.partial import PartialClusterOps
from relevanceai.operations.cluster.cluster import ClusterOps
from relevanceai._api import APIClient
from relevanceai.operations.cluster.utils import _ClusterOps


class _SubClusterOps(ClusterOps):
    """This class is an intermediate layer between cluster ops and subclusterops.
    It it used to over-write parts of clusterops.
    """

    def __init__(
        self,
        credentials,
        alias: str,
        model,
        dataset: Any,
        vector_fields: list,
        parent_field: str,
        outlier_value=-1,
        outlier_label="outlier",
        **kwargs,
    ):
        """
        Sub Cluster Ops
        """

        # self.dataset = dataset
        self.vector_fields = vector_fields
        self.parent_field = parent_field
        self.credentials = credentials
        self.model = model
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label
        if isinstance(dataset, str):
            self.dataset_id = dataset
        else:
            if hasattr(dataset, "dataset_id"):
                self.dataset_id = dataset.dataset_id
            # needs to have the dataset_id attribute

        if "package" not in self.__dict__:
            self.package = self._get_package(self.model)

        self.model_name = None
        self.model = self._get_model(model)
        self.alias = self._get_alias(alias)
        super().__init__(
            credentials=credentials,
            model=self.model,
            vector_fields=self.vector_fields,
            alias=self.alias,
            dataset_id=self.dataset_id,
            outlier_value=self.outlier_value,
            outlier_label=self.outlier_label,
        )

    def operate(
        self,
        dataset_id: Optional[Union[str, Any]] = None,
        parent_field: str = None,
        vector_fields: Optional[List[str]] = None,
        filters: list = None,
        show_progress_bar: bool = True,
        verbose: bool = True,
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

        if parent_field is None:
            parent_field = self.parent_field

        filters = [] if filters is None else filters

        if not isinstance(dataset_id, str):
            if hasattr(dataset_id, "dataset_id"):
                dataset_id = dataset_id.dataset_id  # type: ignore

        if vector_fields is None:
            vector_fields = self.vector_fields

        self.dataset_id = dataset_id
        if vector_fields is not None:
            vector_field = vector_fields[0]
            self.vector_field = vector_field

        # get all documents
        documents = self._get_all_documents(
            dataset_id=dataset_id,
            select_fields=vector_fields + [parent_field],
            show_progress_bar=show_progress_bar,
            include_vector=True,
            filters=filters,
        )

        # fit model, predict and label all documents
        centroid_documents, labelled_documents = self._fit_predict(
            documents=documents,
            vector_field=vector_field,
        )

        results = self._update_documents(
            dataset_id=dataset_id,
            documents=labelled_documents,
            show_progress_bar=show_progress_bar,
        )

        self._insert_centroids(
            dataset_id=dataset_id,
            vector_field=vector_field,
            centroid_documents=centroid_documents,
        )

        # link back to dashboard
        if verbose:
            self._print_app_link()

    def _format_sub_labels(self, parent_values: list, labels: np.ndarray) -> List[str]:
        if len(parent_values) != len(labels):
            raise ValueError("Improper logic for parent values")

        labels = labels.flatten().tolist()
        cluster_labels = [
            label + "-" + str(labels[i])
            if label != self.outlier_value
            else self.outlier_label
            for i, label in enumerate(parent_values)
        ]
        return cluster_labels

    def _get_centroid_documents(
        self, vectors: np.ndarray, labels: List[str]
    ) -> List[Dict[str, Any]]:
        centroid_documents = []

        centroids: Dict[str, Any] = {}
        for label, vector in zip(labels, vectors.tolist()):
            if label not in centroids:
                centroids[label] = []
            centroids[label].append(vector)

        for centroid, vectors in centroids.items():
            centroid_vector = np.array(vectors).mean(0).tolist()
            centroid_document = dict(
                _id=centroid,
                centroid_vector=centroid_vector,
            )
            centroid_documents.append(centroid_document)

        return centroid_documents

    def _insert_centroids(
        self,
        dataset_id: str,
        vector_field: str,
        centroid_documents: List[Dict[str, Any]],
    ) -> None:
        self.services.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centroid_documents,
            vector_fields=[vector_field],
            alias=self.alias,
        )

    def _fit_predict(
        self, documents: List[Dict[str, Any]], vector_field: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        doc_subset = [doc for doc in documents if self.is_field(vector_field, doc)]
        vectors = np.array(
            [self.get_field(vector_field, document) for document in doc_subset]
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

        elif self.package == "sentencetransformers":
            # TODO: make teh config here better
            labels = self.model(vectors)
            labels = np.array(labels)

        elif self.package == "custom":
            labels = self.model.fit_predict(vectors)
            if not isinstance(labels, np.ndarray):
                raise ValueError("Custom model must output np.ndarray for labels")

            if labels.shape[0] != vectors.shape[0]:
                raise ValueError("incorrect shape")

        parent_values = self.get_field_across_documents(self.parent_field, doc_subset)
        labels = self._format_sub_labels(parent_values, labels)

        cluster_field = f"_cluster_.{vector_field}.{self.alias}"
        self.set_field_across_documents(
            field=cluster_field, values=labels, docs=doc_subset
        )

        centroid_documents = self._get_centroid_documents(vectors, labels)

        return centroid_documents, documents


class SubClusterOps(_SubClusterOps, _ClusterOps):  # type: ignore
    def __init__(
        self,
        credentials,
        alias: str,
        dataset,
        model,
        vector_fields: List[str],
        parent_field: str,
        outlier_value=-1,
        outlier_label="outlier",
        **kwargs,
    ):
        """
        Sub Cluster Ops
        """
        self.alias = alias
        # self.dataset = dataset
        self.vector_fields = vector_fields
        self.parent_field = parent_field
        self.credentials = credentials
        self.model = model
        self.outlier_value = outlier_value
        self.outlier_label = outlier_label
        if isinstance(dataset, str):
            self.dataset_id: str = dataset
        else:
            if hasattr(dataset, "dataset_id"):
                self.dataset_id = dataset.dataset_id
            # needs to have the dataset_id attribute

        super().__init__(
            credentials=credentials,
            alias=alias,
            model=model,
            dataset=dataset,
            vector_fields=vector_fields,
            parent_field=parent_field,
        )

    def __call__(self, *args, **kw):
        return self.fit_predict(*args, **kw)

    def fit_predict(
        self,
        dataset,
        vector_fields: List[Any],
        parent_field: str = None,
        filters: Optional[List] = None,
        verbose: bool = False,
    ):
        """

        Run subclustering on your dataset using an in-memory clustering algorithm.

        Parameters
        --------------

        dataset: Dataset
            The dataset to create
        vector_fields: List
            The list of vector fields to run fitting, prediction and updating on
        filters: Optional[List]
            The list of filters to run clustering on
        verbose: bool
            If True, this should be verbose

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()

            from relevanceai.package_utils.datasets import mock_documents
            ds = client.Dataset("sample")

            # Creates 100 sample documents
            documents = mock_documents(100)
            ds.upsert_documents(documents)

            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=10)
            clusterer = ClusterOps(alias="minibatchkmeans-10", model=model)
            clusterer.subcluster_predict_update(
                dataset=ds,
            )

        """
        if self.model is None:
            raise ValueError("No model is detected.")
        filters = [] if filters is None else filters

        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )

        # self.parent_field = parent_field

        # self._init_dataset(dataset)
        self.vector_fields = vector_fields  # type: ignore

        # make sure to only get fields where vector fields exist
        filters += [
            {
                "field": f,
                "filter_type": "exists",
                "condition": "==",
                "condition_value": " ",
            }
            for f in vector_fields  # type: ignore
        ]

        if verbose:
            print("Fitting and predicting on all documents")
        # Here we run subfitting on these documents

        clustered_docs = self.subcluster_predict_documents(
            vector_fields=vector_fields, filters=filters, verbose=False
        )

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )

        # Store subcluster in the metadata
        if parent_field is None:
            parent_field = self.parent_field

        self.store_subcluster_metadata(
            parent_field=parent_field, cluster_field=self._get_cluster_field_name()
        )

    def store_subcluster_metadata(self, parent_field: str, cluster_field: str):
        """
        Store subcluster metadata
        """
        return self.append_metadata_list(
            field="_subcluster_",
            value_to_append={
                "parent_field": parent_field,
                "cluster_field": cluster_field,
            },
        )

    def subpartialfit_predict_update(
        self,
        dataset,
        vector_fields: list,
        filters: Optional[list] = None,
        cluster_ids: Optional[list] = None,
        verbose: bool = True,
    ):
        """

        Run partial fit subclustering on your dataset.

        Parameters
        ------------

        dataset: Dataset
            The dataset to call fit predict update on
        vector_fields: list
            The list of vector fields
        filters: list
            The list of filters

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()

            from relevanceai.package_utils.datasets import mock_documents
            ds = client.Dataset("sample")
            # Creates 100 sample documents
            documents = mock_documents(100)
            ds.upsert_documents(documents)

            from sklearn.cluster import MiniBatchKMeans
            model = MiniBatchKMeans(n_clusters=10)
            clusterer = ClusterOps(alias="minibatchkmeans-10", model=model)
            clusterer.subpartialfit_predict_update(
                dataset=ds,
            )

        """
        # Get data

        # self._init_dataset(dataset)

        filters = [] if filters is None else filters
        cluster_ids = [] if cluster_ids is None else cluster_ids
        # Loop through each unique cluster ID and run clustering
        parent_field = self.parent_field

        print("Getting unique cluster IDs...")
        unique_clusters = self.list_unique(parent_field)

        for i, unique_cluster in enumerate(tqdm(unique_clusters)):
            cluster_filters = filters.copy()
            cluster_filters += [
                {
                    "field": parent_field,
                    "filter_type": "category",
                    "condition": "==",
                    "condition_value": unique_cluster,
                }
            ]
            self.partial_fit_predict_update(
                dataset=self.dataset_id,
                vector_fields=vector_fields,
                filters=cluster_filters,
                verbose=False,
            )

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )

    def list_unique(
        self,
        field: str = None,
        minimum_amount: int = 3,
        dataset_id: str = None,
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
            cluster_ops.list_unique()

        Parameters
        -------------
        alias: str
            The alias to use for clustering
        minimum_cluster_size: int
            The minimum size of the clusters
        dataset_id: str
            The dataset ID
        num_clusters: int
            The number of clusters

        """
        # Mainly to be used for subclustering
        # Get the cluster alias
        if dataset_id is None:
            self._check_for_dataset_id()
            dataset_id = self.dataset_id

        # currently the logic for facets is that when it runs out of pages
        # it just loops - therefore we need to store it in a simple hash
        # and then add them to a list
        all_cluster_ids: Set = set()

        while len(all_cluster_ids) < num_clusters:
            facet_results = self.datasets.facets(
                dataset_id=dataset_id,
                fields=[field],
                page_size=100,
                page=1,
                asc=True,
            )
            if "results" in facet_results:
                facet_results = facet_results["results"]
            if field not in facet_results:
                raise ValueError("Not valid field. Check schema.")
            for facet in facet_results[field]:
                if facet["frequency"] > minimum_amount:
                    curr_len = len(all_cluster_ids)
                    all_cluster_ids.add(facet[field])
                    new_len = len(all_cluster_ids)
                    if new_len == curr_len:
                        return list(all_cluster_ids)

        return list(all_cluster_ids)

    def subcluster_predict_documents(
        self,
        vector_fields: Optional[List] = None,
        filters: Optional[List] = None,
        cluster_ids: Optional[List] = None,
        verbose: bool = True,
    ):
        """
        Subclustering using fit predict update. This will loop through all of the
        different clusters and then run subclustering on them. For this, you need to

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("sample")

            # Creating 100 sample documents
            from relevanceai.package_utils.datasets import mock_documents
            documents = mock_documents(100)
            ds.upsert_documents(documents)

            # Run simple clustering first
            ds.auto_cluster("kmeans-3", vector_fields=["sample_1_vector_"])

            # Start KMeans
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=20)

            # Run subclustering.
            cluster_ops = client.ClusterOps(
                alias="subclusteringkmeans",
                model=model,
                parent_alias="kmeans-3")


        """
        filters = [] if filters is None else filters
        cluster_ids = [] if cluster_ids is None else cluster_ids
        # Loop through each unique cluster ID and run clustering

        if verbose:
            print("Getting unique cluster IDs...")
        if not cluster_ids:
            unique_clusters = self.list_unique(self.parent_field)
        else:
            unique_clusters = cluster_ids

        self._list_of_cluster_ops = []

        for i, unique_cluster in enumerate(tqdm(unique_clusters)):
            cluster_filters = filters.copy()
            cluster_filters += [
                {
                    "field": self.parent_field,
                    "filter_type": "category",
                    "condition": "==",
                    "condition_value": unique_cluster,
                }
            ]
            # Create a ClusterOps object with the filter
            if vector_fields is None:
                vector_fields = self.vector_fields
            ops = _SubClusterOps(
                credentials=self.credentials,
                model=self.model,
                vector_fields=vector_fields,
                alias=self.alias,
                dataset=self.dataset_id,
                parent_field=self.parent_field,
                n_clusters=None,
                cluster_config=None,
                outlier_value=self.outlier_value,
                outlier_label=self.outlier_label,
            )
            ops.operate(
                dataset_id=self.dataset_id,
                vector_fields=vector_fields,
                filters=cluster_filters,
                verbose=False,
            )
            self._list_of_cluster_ops.append(ops)

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
