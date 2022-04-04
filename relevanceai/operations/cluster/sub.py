"""
SubClustering Operations

Sub Clustering allows users to define subclusters.

"""
from typing import Optional, List, Any
from tqdm.auto import tqdm
from relevanceai.operations.cluster.partial import PartialClusterOps
from relevanceai.operations.cluster.cluster import ClusterOps
from relevanceai._api import APIClient


class SubClusterOps(PartialClusterOps):
    def __init__(
        self,
        credentials,
        alias,
        dataset,
        model,
        vector_fields: list,
        parent_field: str,
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
        if isinstance(dataset, str):
            self.dataset_id = dataset
        else:
            if hasattr(dataset, "dataset_id"):
                self.dataset_id = dataset.dataset_id
            # needs to have the dataset_id attribute

        super().__init__(credentials=credentials)

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

        # Updating the db
        # print("Updating the database...")
        # results = self._update_documents(
        #     self.dataset_id, clustered_docs, chunksize=10000
        # )
        # self.logger.info(results)

        # # Update the centroid collection
        # self.model.vector_fields = vector_fields

        # self._insert_centroid_documents()

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
            ops = ClusterOps(
                credentials=self.credentials,
                model=self.model,
                vector_fields=vector_fields,
                alias=None,
                dataset_id=None,
                n_clusters=None,
                cluster_config=None,
                outlier_value=-1,
                outlier_label="outlier",
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
