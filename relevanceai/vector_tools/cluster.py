# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

from typing import List, Union, Dict, Any, Optional
from doc_utils import DocUtils
from joblib.memory import Memory

from relevanceai.api.client import BatchAPIClient
from relevanceai.logger import LoguruLogger
from relevanceai.vector_tools.constants import CLUSTER, CLUSTER_DEFAULT_ARGS
from relevanceai.errors import ClusteringResultsAlreadyExistsError
from relevanceai.vector_tools.cluster_evaluate import ClusterEvaluate


class ClusterBase(LoguruLogger, DocUtils):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, vectors):
        """ """
        raise NotImplementedError

    def _concat_vectors_from_list(self, list_of_vectors: list):
        """Concatenate 2 vectors together in a pairwise fashion"""
        return [np.concatenate(x) for x in list_of_vectors]

    def fit_documents(
        self,
        vector_fields: list,
        docs: list,
        alias: str = "default",
        cluster_field: str = "_cluster_",
        return_only_clusters: bool = True,
        inplace: bool = True,
    ):
        """
        Train clustering algorithm on documents and then store the labels
        inside the documents.

        Parameters
        -----------
        vector_field: list
            The vector field of the documents
        docs: list
            List of documents to run clustering on
        alias: str
            What the clusters can be called
        cluster_field: str
            What the cluster fields should be called
        return_only_clusters: bool
            If True, return only clusters, otherwise returns the original document
        inplace: bool
            If True, the documents are edited inplace otherwise, a copy is made first
        kwargs: dict
            Any other keyword argument will go directly into the clustering algorithm

        """
        if len(vector_fields) == 1:
            # filtering out entries not containing the specified vector
            docs = list(filter(DocUtils.list_doc_fields, docs))
            vectors = self.get_field_across_documents(
                vector_fields[0], docs, missing_treatment="skip"
            )
        else:
            # In multifield clusering, we get all the vectors in each document
            # (skip if they are missing any of the vectors)
            # Then run clustering on the result
            docs = list(self.filter_docs_for_fields(vector_fields, docs))
            all_vectors = self.get_fields_across_documents(
                vector_fields, docs, missing_treatment="skip_if_any_missing"
            )
            vectors = self._concat_vectors_from_list(all_vectors)

        cluster_labels = self.fit_transform(vectors)

        # Label the clusters
        cluster_labels = self._label_clusters(cluster_labels)

        if isinstance(vector_fields, list):
            set_cluster_field = f"{cluster_field}.{'.'.join(vector_fields)}.{alias}"
        elif isinstance(vector_fields, str):
            set_cluster_field = f"{cluster_field}.{vector_fields}.{alias}"

        if inplace:
            self.set_field_across_documents(
                set_cluster_field,
                cluster_labels,
                docs,
            )
            if return_only_clusters:
                return [
                    {"_id": d.get("_id"), cluster_field: d.get(cluster_field)}
                    for d in docs
                ]
            return docs

        new_docs = docs.copy()

        self.set_field_across_documents(set_cluster_field, cluster_labels, new_docs)

        if return_only_clusters:
            return [
                {"_id": d.get("_id"), cluster_field: d.get(cluster_field)} for d in docs
            ]
        return docs

    def to_metadata(self):
        """You can also store the metadata of this clustering algorithm"""
        raise NotImplementedError

    @property
    def metadata(self):
        return self.to_metadata()

    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "Cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]


class CentroidCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, vectors):
        raise NotImplementedError

    @abstractmethod
    def get_centers(self) -> Union[np.ndarray, List[list]]:
        """Get centers for the centroid-based clusters"""
        raise NotImplementedError

    def get_centroid_docs(self) -> List:
        """Get the centroid documents to store."""
        self.centers = self.get_centers()
        if isinstance(self.centers, np.ndarray):
            self.centers = self.centers.tolist()
        return [
            {"_id": f"cluster_{i}", "centroid_vector_": self.centers[i]}
            for i in range(len(self.centers))
        ]


class DensityCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    def fit_transform(self, vectors):
        raise NotImplementedError


class MiniBatchKMeans(CentroidCluster):
    def __init__(
        self,
        k: Union[None, int] = 10,
        init: str = "k-means++",
        verbose: bool = False,
        compute_labels: bool = True,
        max_no_improvement: int = 2,
    ):
        """
        Kmeans Centroid Clustering

        Parameters
        ------------
        k: int
            The number of clusters
        init: str
            The optional parameter to be clustering
        verbose: bool
            If True, will print what is happening
        compute_labels: bool
            If True, computes the labels of the cluster
        max_no_improvement: int
            The maximum number of improvemnets
        """
        self.k = k
        self.init = init
        self.verbose = verbose
        self.compute_labels = compute_labels
        self.max_no_improvement = max_no_improvement

    def _init_model(self):
        from sklearn.cluster import MiniBatchKMeans

        self.km = MiniBatchKMeans(
            n_clusters=self.k,
            init=self.init,
            verbose=self.verbose,
            compute_labels=self.compute_labels,
            max_no_improvement=self.max_no_improvement,
        )
        return

    def fit_transform(self, vectors: Union[np.ndarray, List]):
        """
        Fit and transform transform the vectors
        """
        if not hasattr(self, "km"):
            self._init_model()
        self.km.fit(vectors)
        cluster_labels = self.km.labels_.tolist()
        # cluster_centroids = km.cluster_centers_
        return cluster_labels

    def get_centers(self):
        """Returns centroids of clusters"""
        return [list(i) for i in self.km.cluster_centers_]

    def to_metadata(self):
        """Editing the metadata of the function"""
        return {
            "k": self.k,
            "init": self.init,
            "verbose": self.verbose,
            "compute_labels": self.compute_labels,
            "max_no_improvement": self.max_no_improvement,
            "number_of_clusters": self.k,
        }


# class KMedoids(CentroidCluster):
#     def fit_transform(self,
#         vectors: np.ndarray,
#         cluster_args: Optional[Dict[Any, Any]] = CLUSTER_DEFAULT_ARGS['kmedoids'],
#         k: Union[None, int] = 10,
#     ) -> np.ndarray:
#         try:
#             from sklearn_extra.cluster import KMedoids
#         except ModuleNotFoundError as e:
#             raise ModuleNotFoundError(
#                 f"{e}\nInstall umap\n \
#                 pip install -U relevanceai[kmedoids]"
#             )
#         self.logger.debug(f"{cluster_args}")
#         km = KMedoids(n_clusters=k, **cluster_args).fit(vectors)
#         cluster_labels = km.labels_
#         # cluster_centroids = km.cluster_centers_
#         return cluster_labels


class KMeans(MiniBatchKMeans):
    def __init__(
        self,
        k=10,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.n_clusters = k

    def _init_model(self):
        from sklearn.cluster import KMeans

        self.km = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            verbose=self.verbose,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
        )
        return

    def to_metadata(self):
        """Editing the metadata of the function"""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "copy_x": self.copy_x,
            "algorithm": self.algorithm,
        }


class HDBSCANClusterer(DensityCluster):
    def __init__(
        self,
        algorithm: str = "best",
        alpha: float = 1.0,
        approx_min_span_tree: bool = True,
        gen_min_span_tree: bool = False,
        leaf_size: int = 40,
        memory=Memory(cachedir=None),
        metric: str = "euclidean",
        min_samples=None,
        p=None,
        min_cluster_size: Union[None, int] = 10,
    ):
        self.algorithm = algorithm
        self.alpha = alpha
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.leaf_size = leaf_size
        self.memory = memory
        self.metric = metric
        self.min_samples = min_samples
        self.p = p
        self.min_cluster_size = min_cluster_size

    def fit_transform(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        try:
            from hdbscan import HDBSCAN
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall hdbscan\n \
                pip install -U relevanceai[hdbscan]"
            )
        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            algorithm=self.algorithm,
            approx_min_span_tree=self.approx_min_span_tree,
            gen_min_span_tree=self.gen_min_span_tree,
            leaf_size=self.leaf_size,
            memory=self.memory,
            metric=self.metric,
            min_samples=self.min_samples,
            p=self.p,
        ).fit(vectors)
        cluster_labels = hdbscan.labels_
        return cluster_labels


class Cluster(ClusterEvaluate, BatchAPIClient, ClusterBase):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    @staticmethod
    def _choose_k(vectors: np.ndarray):
        """ "
        Choose k clusters
        """
        # Partitioning methods
        # if check_type(self.cluster, CLUSTER_NUMERIC):
        """
        Scaled_inertia = inertia(k)/inertia(k=1) + (a * K)
        where a is penalty factor of num_clusters
        """
        return 10

    @staticmethod
    def cluster(
        vectors: np.ndarray,
        cluster: Union[CLUSTER, ClusterBase],
        cluster_args: Union[None, dict],
        k: Union[None, int] = None,
    ) -> np.ndarray:
        """
        Cluster vectors
        """
        if isinstance(cluster, str):
            if cluster_args is None:
                cluster_args = CLUSTER_DEFAULT_ARGS[cluster]
            if cluster in ["kmeans", "kmedoids"]:
                if (k is None and cluster_args is None) or (
                    "n_clusters" not in cluster_args.keys()
                ):
                    k = Cluster._choose_k(vectors)
                if cluster == "kmeans":
                    return KMeans(k=k, **cluster_args).fit_transform(vectors=vectors)
                elif cluster == "kmedoids":
                    raise NotImplementedError
                    # return KMedioids().fit_transform(vectors=vectors, cluster_args=cluster_args)
            elif cluster == "hdbscan":
                return HDBSCANClusterer(**cluster_args).fit_transform(vectors=vectors)

        elif isinstance(cluster, ClusterBase):
            return cluster().fit_transform(vectors=vectors, cluster_args=cluster_args)
        raise ValueError("Not valid cluster input.")

    def kmeans_cluster(
        self,
        dataset_id: str,
        vector_fields: list,
        filters: List = [],
        k: Union[None, int] = 10,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = True,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "auto",
        alias: str = None,
        cluster_field: str = "_cluster_",
        update_documents_chunksize: int = 50,
        overwrite: bool = False,
    ):
        """
        This function performs all the steps required for Kmeans clustering:
        1- Loads the data
        2- Clusters the data
        3- Updates the data with clustering info
        4- Adds the centroid to the hidden centroid collection

        Parameters
        ----------
        dataset_id : string
            name of the dataser
        vector_fields : list
            a list containing the vector field to be used for clustering
        filters : list
            a list to filter documents of the dataset,
        k : int
            K in Kmeans
        init : string
            "k-means++" -> Kmeans algorithm parameter
        n_init : int
            number of reinitialization for the kmeans algorithm
        max_iter : int
            max iteration in the kmeans algorithm
        tol : int
            tol in the kmeans algorithm
        verbose : bool
            True by default
        random_state = None
            None by default -> Kmeans algorithm parameter
        copy_x : bool
            True bydefault
        algorithm : string
            "auto" by default
        alias : string
            "kmeans", string to be used in naming of the field showing the clustering results
        cluster_field: string
            "_cluster_", string to name the main cluster field
        overwrite : bool
            False by default, To overwite an existing clusering result

        Example
        -------------

        >>> client.vector_tools.cluster.kmeans_cluster(
            dataset_id="sample_dataset",
            vector_fields=["sample_1_vector_"] # Only 1 vector field is supported for now
        )
        """

        if alias is None:
            alias = "kmeans_" + str(k)

        EXPECTED_CLUSTER_OUTFIELD = ".".join([cluster_field, vector_fields[0], alias])
        if (
            EXPECTED_CLUSTER_OUTFIELD in self.datasets.schema(dataset_id)
            and not overwrite
        ):
            raise ClusteringResultsAlreadyExistsError(EXPECTED_CLUSTER_OUTFIELD)

        filters = filters + [
            {
                "field": vector_fields[0],
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
        ]
        # load the documents
        self.logger.warning(
            "Retrieving documents... This can take a while if the dataset is large."
        )
        docs = self.get_all_documents(
            dataset_id=dataset_id, filters=filters, select_fields=vector_fields
        )

        # Cluster
        clusterer = KMeans(
            k=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )
        clustered_docs = clusterer.fit_documents(
            vector_fields,
            docs,
            alias=alias,
            cluster_field=cluster_field,
            return_only_clusters=True,
            inplace=False,
        )

        # Updating the db
        results = self.update_documents(
            dataset_id, clustered_docs, chunksize=update_documents_chunksize
        )
        self.logger.info(results)

        # Update the centroid collection
        centers = clusterer.get_centroid_docs()
        results = self.services.cluster.centroids.insert(
            dataset_id=dataset_id,
            cluster_centers=centers,
            vector_field=vector_fields[0],
            alias=alias,
        )
        self.logger.info(results)

        return centers

    def hdbscan_cluster(
        self,
        dataset_id: str,
        vector_fields: list,
        filters: List = [],
        algorithm: str = "best",
        alpha: float = 1.0,
        approx_min_span_tree: bool = True,
        gen_min_span_tree: bool = False,
        leaf_size: int = 40,
        memory=Memory(cachedir=None),
        metric: str = "euclidean",
        min_samples=None,
        p=None,
        min_cluster_size: Union[None, int] = 10,
        alias: str = "hdbscan",
        cluster_field: str = "_cluster_",
        update_documents_chunksize: int = 50,
        overwrite: bool = False,
    ):
        """
        This function performs all the steps required for hdbscan clustering:
        1- Loads the data
        2- Clusters the data
        3- Updates the data with clustering info
        4- Adds the centroid to the hidden centroid collection

        Parameters
        ----------
        dataset_id : string
            name of the dataser
        vector_fields : list
            a list containing the vector field to be used for clustering
        filters : list
            a list to filter documents of the dataset
        algorithm : str
            hdbscan configuration parameter default to "best"
        alpha: float
            hdbscan configuration parameter default to 1.0
        approx_min_span_tree: bool
            hdbscan configuration parameter default to True
        gen_min_span_tree: bool
            hdbscan configuration parameter default to False
        leaf_size: int
            hdbscan configuration parameter default to 40
        memory = Memory(cachedir=None)
            hdbscan configuration parameter on memory management
        metric: str = "euclidean"
            hdbscan configuration parameter default to "euclidean"
        min_samples = None
            hdbscan configuration parameter default to None
        p = None
            hdbscan configuration parameter default to None
        min_cluster_size:
            minimum cluster size, 10 by default
        alias : string
            "hdbscan", string to be used in naming of the field showing the clustering results
        cluster_field: string
            "_cluster_", string to name the main cluster field
        overwrite : bool
            False by default, To overwite an existing clusering result

        Example
        -------------

        >>> client.vector_tools.cluster.hdbscan_cluster(
            dataset_id="sample_dataset",
            vector_fields=["sample_1_vector_"] # Only 1 vector field is supported for now
        )
        """

        if (
            ".".join([cluster_field, vector_fields[0], alias])
            in self.datasets.schema(dataset_id)
            and not overwrite
        ):
            raise ClusteringResultsAlreadyExistsError(
                ".".join([cluster_field, vector_fields[0], alias])
            )
        # load the documents
        docs = self.get_all_documents(
            dataset_id=dataset_id, filters=filters, select_fields=vector_fields
        )

        # get vectors
        if len(vector_fields) > 1:
            raise ValueError(
                "We currently do not support more than 1 vector field yet. This will be supported in the future."
            )

        # Cluster
        clusterer = HDBSCANClusterer(
            algorithm=algorithm,
            alpha=alpha,
            approx_min_span_tree=approx_min_span_tree,
            gen_min_span_tree=gen_min_span_tree,
            leaf_size=leaf_size,
            memory=memory,
            metric=metric,
            min_samples=min_samples,
            p=p,
            min_cluster_size=min_cluster_size,
        )
        clustered_docs = clusterer.fit_documents(
            vector_fields, docs, alias=alias, return_only_clusters=True
        )

        # Updating the db
        # formatted_clustered_docs = [
        #     {cluster_field:{vector_fields[0]:{alias:res}},
        #     '_id':docs[i]['_id']}
        #     for i,res in enumerate(clustered_docs)]
        results = self.update_documents(
            dataset_id, clustered_docs, chunksize=update_documents_chunksize
        )
        self.logger.info(results)
        return clustered_docs
