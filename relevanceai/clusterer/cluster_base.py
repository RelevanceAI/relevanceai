"""Document utilities
"""
import numpy as np
from doc_utils import DocUtils
from abc import abstractmethod, ABC
from typing import Union

class ClusterBase(DocUtils, ABC):
    """
    Create an ABC
    """
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
        self.vector_fields = vector_fields
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

    @abstractmethod
    def to_metadata(self):
        """You can also store the metadata of this clustering algorithm"""
        raise NotImplementedError

    @property
    def metadata(self):
        return self.to_metadata()

    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]
