# -*- coding: utf-8 -*-
"""KMeans Clustering
"""
import numpy as np
import warnings
from typing import Union, List, Optional

from relevanceai.clusterer.clusterer import ClusterOps
from relevanceai.clusterer.cluster_base import ClusterBase
from relevanceai.dataset_api import Dataset


class KMeansModel(ClusterBase):
    """
    Simple K means model powered by Scikit Learn.

    Run KMeans Clustering.

    Parameters
    ------------
    alias: str
        The name to call your cluster.  This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
    k: str
        The number of clusters in your K Means
    cluster_field: str
        The field from which to store the cluster. This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}

    You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Example
    -----------

    .. code-block::

        from relevanceai import Client
        client = Client()
        dataset_id = "_github_repo_vectorai"
        df = client.Dataset(dataset_id)

        from relevanceai.clusterer import KMeansModel
        model = KMeansModel(k=3)

        cluster_ops = client.ClusterOps(model=model, alias="kmeans")
        cluster_ops.fit(df, vector_fields=["documentation_vector_"])

    """

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

    def fit_predict(self, vectors: Union[np.ndarray, List]):
        """
        Fit and transform transform the vectors
        """
        if not hasattr(self, "km"):
            self._init_model()
        self.km.fit(vectors)
        cluster_labels = self.km.labels_.tolist()
        # cluster_centroids = km.cluster_centers_
        return cluster_labels

    @property
    def metadata(self):
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

    def get_centers(self):
        """Returns centroids of clusters"""
        if not hasattr(self, "vector_fields") or len(self.vector_fields) == 1:
            return [list(i) for i in self.km.cluster_centers_]

        # Returning for multiple vector fields
        cluster_centers = []
        for i, center in enumerate(self.km.cluster_centers_):
            cluster_center_doc = {}
            for j, vf in enumerate(self.vector_fields):
                deconcat_center = center[
                    self._vector_field_length[vf]["start"] : self._vector_field_length[
                        vf
                    ]["end"]
                ].tolist()
                cluster_center_doc[vf] = deconcat_center
            cluster_centers.append(cluster_center_doc.copy())
        return cluster_centers

    def get_centroid_documents(self) -> List:
        """
        Get the centroid documents to store.
        If single vector field returns this:

        .. code-block::

            {
                "_id": "document-id-1",
                "centroid_vector_": [0.23, 0.24, 0.23]
            }

        If multiple vector fields returns this returns multiple:

        .. code-block::

            {
                "_id": "document-id-1",
                "blue_vector_": [0.12, 0.312, 0.42],
                "red_vector_": [0.23, 0.41, 0.3]
            }


        """
        self.centers = self.get_centers()
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


class KMeansClusterOps(ClusterOps):
    """
    Run KMeans Clustering.

    Parameters
    ----------
    alias: str
        The name to call your cluster.  This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}
    k: str
        The number of clusters in your K Means
    cluster_field: str
        The field from which to store the cluster. This will be used to store your clusters in the form of {cluster_field{.vector_field.alias}

    You can read about the other parameters here: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Example
    -----------

    >>> from relevanceai import Client
    >>> client = Client()
    >>>
    >>> clusterer = client.KMeansClusterOps(alias="kmeans_cluster_sample")
    >>> df = client.Dataset("sample")
    >>> clusterer.fit(df, vector_fields=["sample_vector_"])

    """

    def __init__(
        self,
        alias: str,
        project: str,
        api_key: str,
        firebase_uid: str,
        k: Union[None, int] = 10,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "auto",
        cluster_field: str = "_cluster_",
    ):
        self.model: KMeansModel = KMeansModel(
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
        super().__init__(
            model=self.model,
            alias=alias,
            cluster_field=cluster_field,
            project=project,
            api_key=api_key,
            firebase_uid=firebase_uid,
        )
        warnings.warn("Function has been deprecated.", DeprecationWarning)

    def _insert_centroid_documents(self):
        if hasattr(self.model, "get_centroid_documents"):
            centers = self.model.get_centroid_documents()

            # Change centroids insertion
            results = self.services.cluster.centroids.insert(
                dataset_id=self.dataset_id,
                cluster_centers=centers,
                vector_fields=self.vector_fields,
                alias=self.alias,
            )
            self.logger.info(results)

            self.datasets.cluster.centroids.list_closest_to_center(
                self.dataset_id,
                vector_fields=self.vector_fields,
                alias=self.alias,
                centroid_vector_fields=self.vector_fields,
                page_size=20,
            )
        return

    def fit(
        self, dataset: Union[Dataset, str], vector_fields: List, filters: list = []
    ):
        """
        Train clustering algorithm on documents and then store the labels
        inside the documents.

        Parameters
        -----------
        dataset: Union[str, Dataset]
            The dataset to fit it. If string, it will create a dataset
        vector_field: list
            The vector field of the documents
        """
        self.fit_dataset(dataset, vector_fields=vector_fields, filters=filters)
        return self._insert_centroid_documents()
