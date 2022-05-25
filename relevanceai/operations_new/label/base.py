"""
Labelling performs a vector search on the labels and fetches the closest
max_number_of_labels.

"""
from copy import deepcopy
from typing import Any, Dict, List

from relevanceai.operations_new.base import OperationBase


class LabelBase(OperationBase):
    def __init__(
        self,
        vector_field: str,
        expanded: bool = True,
        max_number_of_labels: int = 1,
        similarity_metric: str = "cosine",
        similarity_threshold: float = 0.1,
        label_field="label",
        label_vector_field="label_vector_",
        **kwargs,
    ):
        self.vector_field = vector_field
        self.expanded = expanded
        self.max_number_of_labels = max_number_of_labels
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.label_field = label_field
        self.label_vector_field = label_vector_field

        for k, v in kwargs.items():
            setattr(self, k, v)

    def transform(  # type: ignore
        self,
        documents,
        label_documents,
    ) -> List[Dict[str, Any]]:

        """Get all vectors, search across

        Parameters
        ----------
        documents
            the documents to be labeled
        label_documents
            The documents that contain the labels.

        Example
        -------
        .. code-block::

            ds = client.Dataset(...)
            # label an entire dataset
            ds.label(
                vector_field="sample_1_vector_",
                label_documents=[
                    {
                        "label": "value",
                        "price": 0.3,
                        "label_vector_": [1, 1, 1]
                    },
                    {
                        "label": "value-2",
                        "label_vector_": [2, 1, 1]
                    },
                ],
                expanded=True # stored as dict or list
            )

        If missing "label", returns Error - labels missing `label` field
        writes loop to set `label` field

        If you want all values in a label document plus similarity, you need to set
        expanded=True

        Returns
        -------
            A list of dictionaries.

        """

        # Get all vectors
        vectors = self.get_field_across_documents(self.vector_field, documents)
        for i, vector in enumerate(vectors):
            # search across
            labels = self._get_nearest_labels(
                vector=vector,
                label_documents=label_documents,
            )
            # TODO: add inplace=True
            self.set_field("_label_", documents[i], labels)
        return documents

    @property
    def name(self):
        return "labelling"

    def _get_nearest_labels(
        self,
        vector: List[float],
        label_documents: List[Dict[str, Any]],
    ):
        """It takes a vector and a list of documents, and returns a list of labels

        Parameters
        ----------
        vector : List[float]
            List[float]
        label_documents : List[Dict[str, Any]]
            List[Dict[str, Any]]

        Returns
        -------
            The return value is a list of labels.

        """
        # perform cosine similarity
        if self.similarity_metric == "cosine":
            labels = self.cosine_similarity(
                query_vector=vector,
                vector_field=self.label_vector_field,
                documents=label_documents,
                max_number_of_labels=self.max_number_of_labels,
                similarity_threshold=self.similarity_threshold,
            )
        else:
            raise ValueError(
                "Only cosine similarity metric is supported at the moment."
            )

        # for the label vectors
        if self.expanded:
            return labels
        else:
            # get a list of labels
            return self.get_field_across_documents(self.label_field, labels)

    def cosine_similarity(
        self,
        query_vector,
        vector_field,
        documents,
        reverse=True,
        score_field: str = "_label_score",
        max_number_of_labels: int = 1,
        similarity_threshold: float = 0,
    ):
        """It takes a query vector, a vector field, a list of documents, and a few other parameters, and
        returns a list of documents sorted by their cosine similarity to the query vector

        Parameters
        ----------
        query_vector
            the vector you want to compare against
        vector_field
            the field in the documents that contains the vector
        documents
            list of documents
        reverse, optional
            True/False
        score_field : str, optional
            str = "_label_score"
        max_number_of_labels : int, optional
            int = 1,
        similarity_threshold : float, optional
            float = 0,

        Returns
        -------
            A list of dictionaries.

        """
        from scipy.spatial import distance

        sort_key = [
            1 - distance.cosine(i, query_vector)
            for i in self.get_field_across_documents(vector_field, documents)
        ]
        self.set_field_across_documents(score_field, sort_key, documents)
        labels = sorted(documents, reverse=reverse, key=lambda x: x[score_field])[
            :max_number_of_labels
        ]
        labels = deepcopy(labels)
        # remove labels from labels
        [l.pop(vector_field) for l in labels]
        # TODO: add similarity_threshold
        return labels

    def get_operation_metadata(self) -> Dict[str, Any]:
        return dict(
            operation="label",
            values=str(
                {
                    "vector_fields": self.vector_fields,
                    "expanded": self.expanded,
                    "max_number_of_labels": self.max_number_of_labels,
                    "similarity_metric": self.similarity_metric,
                    "similarity_threshold": self.similarity_threshold,
                    "label_field": self.label_field,
                    "label_vector_field": self.label_vector_field,
                }
            ),
        )
