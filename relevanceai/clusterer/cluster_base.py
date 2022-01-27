import numpy as np
from doc_utils import DocUtils
from abc import abstractmethod, ABC
from typing import Union, List


class ClusterBase(DocUtils, ABC):
    """
    A Cluster Base for models to be inherited.
    """

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, vectors: list) -> List[Union[str, float, int]]:
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

            def fit_transform(self, vectors: Union[np.ndarray, List]):
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
        if isinstance(label, (int, float)):
            return "cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]


class CentroidClusterBase(ClusterBase, ABC):
    @abstractmethod
    def get_centroid_documents(self):
        """Get the centroid documents."""
        pass
