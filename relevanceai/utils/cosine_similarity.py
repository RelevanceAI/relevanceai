"""Cosine similarity operations
"""
from operator import is_
from typing import List, Dict, Any
from relevanceai.utils.integration_checks import is_scipy_available
import numpy as np
from doc_utils import DocUtils


def cosine_similarity(a, b):
    """Cosine similarity utility"""
    if is_scipy_available():
        from scipy import spatial

        return 1 - spatial.distance.cosine(a, b)
    else:
        a_array = np.array(a)
        b_array = np.array(b)
        return a_array.dot(b_array) / (
            np.linalg.norm(a_array, axis=1) * np.linalg.norm(b_array)
        )


def cosine_similarity_across_documents(vector_field, documents):
    """Cosine similarity across documents"""
    pass


def cosine_similarity_matrix(all_vectors):
    from sklearn.metrics import pairwise_distances

    dist_out = 1 - pairwise_distances(all_vectors, metric="cosine")
    return dist_out


def get_cosine_similarity_scores(
    self,
    documents: List[Dict[str, Any]],
    anchor_document: Dict[str, Any],
    vector_field: str,
) -> List[float]:
    """
    Compare scores based on cosine similarity

    Args:
        other_documents:
            List of documents (Python Dictionaries)
        anchor_document:
            Document to compare all the other documents with.
        vector_field:
            The field in the documents to compare
    Example:
        >>> documents = [{...}]
        >>> ViClient.get_cosine_similarity_scores(documents[1:10], documents[0])

    """
    similarity_scores = []
    for i, doc in enumerate(documents):
        similarity_score = self.calculate_cosine_similarity(
            self.get_field(vector_field, doc),
            self.get_field(vector_field, anchor_document),
        )
        similarity_scores.append(similarity_score)
    return similarity_scores
