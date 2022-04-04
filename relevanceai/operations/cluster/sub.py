from typing import Optional, List, Any

from tqdm import tqdm

from relevanceai.operations.cluster.utils import _ClusterOps


class SubClusterOps(_ClusterOps):
    def subcluster_predict_update(
        self,
        dataset,
        vector_fields: List[Any],
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

        self._init_dataset(dataset)
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
            vector_fields=vector_fields, filters=filters, verbose=verbose
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

        self._init_dataset(dataset)

        filters = [] if filters is None else filters
        cluster_ids = [] if cluster_ids is None else cluster_ids
        # Loop through each unique cluster ID and run clustering
        parent_field = self._get_cluster_field_name(self.parent_alias)

        print("Getting unique cluster IDs...")
        unique_clusters = self.unique_cluster_ids(alias=self.parent_alias)

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
        parent_field = self._get_cluster_field_name(self.parent_alias)

        if verbose:
            print("Getting unique cluster IDs...")
        if not cluster_ids:
            unique_clusters = self.unique_cluster_ids(alias=self.parent_alias)
        else:
            unique_clusters = cluster_ids

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
            self.fit_predict_update(
                dataset=self.dataset_id,
                vector_fields=vector_fields,
                filters=cluster_filters,
                include_report=False,
                verbose=False,
            )

        if verbose:
            print(
                "Build your clustering app here: "
                + f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/cluster"
            )
