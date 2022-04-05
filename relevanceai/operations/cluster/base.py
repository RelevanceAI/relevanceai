"""
Custom Cluster Models
-------------------------

The ClusterBase class is intended to be inherited so that users can add their own clustering algorithms 
and models. A cluster base has the following abstractmethods (methods to be overwritten):

- :code:`fit_transform`
- :code:`metadata` (optional if you want to store cluster metadata)
- :code:`get_centers` (optional if you want to store cluster centroid documents)

:code:`CentroidBase` is the most basic class to inherit. Use this class if you have an 
in-memory fitting algorithm.

If your clusters return centroids, you will want to inherit
:code:`CentroidClusterBase`.

If your clusters can fit on batches, you will want to inherit 
:code:`BatchClusterBase`.

If you have both Batches and Centroids, you will want to inherit both.

.. code-block::

    import numpy as np 
    from faiss import Kmeans
    from relevanceai import Client, CentroidClusterBase

    client = Client()
    df = client.Dataset("_github_repo_vectorai")

    class FaissKMeans(CentroidClusterBase):
        def __init__(self, model):
            self.model = model

        def fit_predict(self, vectors):
            vectors = np.array(vectors).astype("float32")
            self.model.train(vectors)
            cluster_labels = self.model.assign(vectors)[1]
            return cluster_labels

        def metadata(self):
            return self.model.__dict__

        def get_centers(self):
            return self.model.centroids

    n_clusters = 10
    d = 512
    alias = f"faiss-kmeans-{n_clusters}"
    vector_fields = ["documentation_vector_"]

    model = FaissKMeans(model=Kmeans(d=d, k=n_clusters))
    clusterer = client.ClusterOps(model=model, alias=alias)
    clusterer.fit_predict_update(dataset=df, vector_fields=vector_fields)
"""

import numpy as np

from doc_utils import DocUtils
from abc import abstractmethod, ABC
from typing import Union, List, Dict, Callable

from relevanceai.utils.integration_checks import (
    is_hdbscan_available,
    is_sklearn_available,
)


class ClusterBase(DocUtils, ABC):
    """
    A Cluster _Base for models to be inherited.
    The most basic class to inherit.
    Use this class if you have an in-memory fitting algorithm.

    If your clusters return centroids, you will want to inherit
    `CentroidClusterBase`.

    If your clusters can fit on batches, you will want to inherit
    `BatchClusterBase`.

    If you have both Batches and Centroids, you will want to inherit both.

    """

    def __call__(self, *args, **kwargs):
        return self.fit_predict(*args, **kwargs)

    @abstractmethod
    def fit_predict(self, vectors: list) -> List[Union[str, float, int]]:
        """Edit this method to implement a ClusterBase.

        Parameters
        -------------
        vectors: list
            The vectors that are going to be clustered

        Example
        ----------

        .. code-block::

            class KMeansModel(ClusterBase):
                def __init__(self, k=10, init="k-means++", n_init=10,
                    max_iter=300, tol=1e-4, verbose=0, random_state=None,
                        copy_x=True,algorithm="auto"):
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
                if not hasattr(self, "km"):
                    self._init_model()
                self.km.fit(vectors)
                cluster_labels = self.km.labels_.tolist()
                # cluster_centroids = km.cluster_centers_
                return cluster_labels

        """
        raise NotImplementedError

    def _concat_vectors_from_list(self, list_of_vectors: list):
        return [np.concatenate(x) for x in list_of_vectors]

    def _get_vectors_from_documents(self, vector_fields, docs):
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
            # Store the vector field lengths to de-concatenate them later
            self._vector_field_length: dict = {}
            prev_vf = 0
            for i, vf in enumerate(self.vector_fields):
                self._vector_field_length[vf] = {}
                self._vector_field_length[vf]["start"] = prev_vf
                end_vf = prev_vf + len(all_vectors[0][i])
                self._vector_field_length[vf]["end"] = end_vf
                # Update the ending
                prev_vf = end_vf

            # Store the vector lengths
            vectors = self._concat_vectors_from_list(all_vectors)
        return vectors

    def __getdoc__(self, documents):
        """What you want in each doc"""
        raise NotImplementedError

    def _bulk_get_doc(self, documents):
        raise NotImplementedError

    def fit_documents(
        self,
        vector_fields: list,
        documents: List[dict],
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
        documents: list
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
        self.vector_fields = vector_fields

        vectors = self._get_vectors_from_documents(vector_fields, documents)

        cluster_labels = self.fit_predict(vectors)

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
                documents,
            )
            if return_only_clusters:
                return [
                    {"_id": d.get("_id"), cluster_field: d.get(cluster_field)}
                    for d in documents
                ]
            return documents

        new_documents = documents.copy()

        self.set_field_across_documents(
            set_cluster_field, cluster_labels, new_documents
        )

        if return_only_clusters:
            return [
                {"_id": d.get("_id"), cluster_field: d.get(cluster_field)}
                for d in documents
            ]
        return documents

    @property
    def metadata(self) -> dict:
        """If metadata is set - this willi be stored on RelevanceAI.
        This is useful when you are looking to compare the metadata of your clusters.
        """
        return {}

    def _label_cluster(self, label: Union[int, str]):
        if not isinstance(label, str):
            return "cluster-" + str(label)
        return label

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]


class AdvancedCentroidClusterBase(ClusterBase, ABC):
    """
    This centroid cluster base assumes that you want to specify
    quite advanced centroid documents.

    You may want to use this if you want to get more control over
    what is actually inserted as a centroid.
    """

    @abstractmethod
    def get_centroid_documents(self) -> List[Dict]:
        """Get the centroid documents."""
        pass


class CentroidBase(ABC):
    """
    Simple centroid base for clusters.
    """

    vector_fields: list
    _label_cluster: Callable

    @abstractmethod
    def get_centers(self) -> List[List[float]]:
        """Add how you need to get centers here. This should return a list of vectors.
        The SDK will then label each center `cluster-0`, `cluster-1`, `cluster-2`, etc... in order.
        If you need more fine-grained control, please see get_centroid_documents.
        """
        pass

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


if is_sklearn_available():
    from sklearn.cluster import KMeans


class SklearnCentroidBase(CentroidBase, ClusterBase):
    def __init__(self, model):
        self.model: KMeans = model

    def get_centers(self):
        if hasattr(self.model, "cluster_centers_"):
            return self.model.cluster_centers_
        # Get the centers for each label
        centers = []
        labels = self.get_unique_labels()
        for l in sorted(np.unique(labels).tolist()):
            # self.model.jkjkj
            centers.append(self._X[self.preds == l].mean(axis=0).tolist())
        return centers

    def fit_predict(self, X):
        self._X = np.array(X)
        self.preds = self.model.fit_predict(X)
        return self.preds

    def get_unique_labels(self):
        if hasattr(self.model, "_labels"):
            labels = self.model._labels
        # Get labels from hdbscan
        elif hasattr(self.model, "labels_"):
            labels = self.model.labels_
        else:
            raise AttributeError(
                "SKLearn has changed labels API - will need to provide way to return cluster centers"
            )
        return sorted(np.unique(labels))

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
        self.centers = self.get_centers()

        if not hasattr(self, "vector_fields") or len(self.vector_fields) == 1:
            if isinstance(self.centers, np.ndarray):
                self.centers = self.centers.tolist()
            centroid_vector_field_name = self.vector_fields[0]
            return [
                {
                    "_id": self._label_cluster(self.model.labels_[i]),
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


if is_hdbscan_available():
    import hdbscan


class HDBSCANClusterBase(SklearnCentroidBase):
    model: "hdbscan.HDBSCAN"

    def get_unique_labels(self):
        return sorted(np.unique(self.model.labels_))

    def get_centers(self):
        labels = self.get_unique_labels()
        centers = []
        for l in labels:
            centers.append(
                self.model._raw_data[self.model.labels_ == l].mean(axis=0).tolist()
            )
        return centers

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
        self.centers = self.get_centers()
        labels = self.get_unique_labels()

        if not hasattr(self, "vector_fields") or len(self.vector_fields) == 1:

            if isinstance(self.centers, np.ndarray):
                self.centers = self.centers.tolist()

            centroid_vector_field_name = self.vector_fields[0]
            return [
                {
                    "_id": self._label_cluster(labels[i]),
                    centroid_vector_field_name: self.centers[i],
                }
                for i in range(len(self.centers))
            ]
        # For one or more vectors, separate out the vector fields
        # centroid documents are created using multiple vector fields
        centroid_docs = []
        for i, c in enumerate(self.centers):
            centroid_doc = {"_id": self._label_cluster(labels[i])}
            for j, vf in enumerate(self.vector_fields):
                centroid_doc[vf] = self.centers[i][vf]
            centroid_docs.append(centroid_doc.copy())
        return centroid_docs


class CentroidClusterBase(ClusterBase, CentroidBase, ABC):
    """
    Inherit this class if you have a centroids-based clustering.
    The difference between this and `Clusterbase` is that you can also additionally
    specify how to get your centers in the
    `get_centers` base. This allows you to store your centers.
    """

    ...


class BatchClusterBase(ClusterBase, ABC):
    """
    Inherit this class if you have a batch-fitting algorithm that needs to be
    trained and then predicted separately.
    """

    @abstractmethod
    def partial_fit(self, vectors):
        """
        Partial fit the vectors.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predict the vectors.
        """
        pass

    def fit_predict(self, X):
        pass
