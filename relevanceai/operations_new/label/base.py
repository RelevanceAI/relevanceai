"""
Labelling performs a vector search on the labels and fetches the closest
max_number_of_labels.

Example
--------

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
    # If missing "label", returns Error - labels missing `label` field
    # writes loop to set `label` field

    # If you want all values in a label document plus similarity, you need to set
    # expanded=True

"""
from typing import Callable, Dict, List, Optional
from doc_utils import DocUtils


class LabelBase(DocUtils):
    def run(
        self,
        vector_field: str,
        documents,
        label_documents,
        expanded: bool = True,
        max_number_of_labels: int = 1,
        similarity_metric: str = "cosine",
        similarity_threshold: float = 0.1,
        label_field="label",
        label_vector_field="label_vector_",
    ):
        """
        For each document, get the vector, match the vector against label vectors, store labels, return
        labelled documents

        Parameters
        ----------
        vector_field : str
            the field in the documents that contains the vector
        documents
            the documents to label
        label_documents
            the documents that contain the labels
        max_number_of_labels : int, optional
            int=1,
        expanded : bool, optional
            if True, then the label vectors are expanded to the same size as the document vectors.
        similarity_metric : str, optional
            str="cosine",
        similarity_threshold : float
            float=0.1,
        label_field, optional
            the field in the label documents that contains the label
        label_vector_field, optional
            the field in the label documents that contains the vector

        Returns
        -------
            A list of documents with the field "_label_" set to the list of labels

        """
        # for each document
        # get vector
        # match vector against label vectors
        # store labels
        # return labelled documents

        # Get all vectors
        vectors = self.get_field_across_documents(vector_field, documents)
        for i, vector in enumerate(vectors):
            # search across
            labels = self._get_nearest_labels(
                vector,
                label_field=label_field,
                label_documents=label_documents,
                expanded=expanded,
                label_vector_field=label_vector_field,
            )
            # TODO: add inplace=True
            self.set_field("_label_", documents[i], labels)
        return documents

    def _get_nearest_labels(
        self,
        vector,
        label_documents,
        label_field: str = "label",
        expanded: bool = True,
        label_vector_field: str = "label_chunkvector_",
        similarity_metric: str = "cosine",
        max_number_of_labels: int = 1,
        similarity_threshold: float = 0.1,
    ):
        # perform cosine similarity
        if similarity_metric == "cosine":
            labels = self.cosine_similarity(
                query_vector=vector,
                vector_field=label_vector_field,
                documents=label_documents,
                max_number_of_labels=max_number_of_labels,
                similarity_threshold=similarity_threshold,
            )
        else:
            raise ValueError(
                "Only cosine similarity metric is supported at the moment."
            )

        # for the label vectors
        if expanded:
            return labels
        else:
            # get a list of labels
            return self.get_field_across_documents(label_field, labels)

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
        from scipy.spatial import distance

        sort_key = [
            1 - distance.cosine(i, query_vector)
            for i in self.get_field_across_documents(vector_field, documents)
        ]
        self.set_field_across_documents(score_field, sort_key, documents)
        labels = sorted(documents, reverse=reverse, key=lambda x: x[score_field])[
            :max_number_of_labels
        ]
        labels = labels.copy()
        # remove labels from labels
        [l.pop(vector_field) for l in labels]
        # TODO: add similarity_threshold
        return labels
