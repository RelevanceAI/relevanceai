from doc_utils.doc_utils import DocUtils
from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from relevanceai.operations.cluster.constants import NEAREST_NEIGHBOURS

doc_utils = DocUtils()


class NearestNeighbours(_Base, DocUtils):
    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    @staticmethod
    def get_nearest_neighbours(
        documents: list,
        vector: list,
        vector_field: str,
        distance_measure_mode: NEAREST_NEIGHBOURS = "cosine",
        callable_distance=None,
        score_field: str = "nearest_neighbour_distance",
    ):
        import scipy.spatial.distance as spatial_distance

        if callable_distance:
            sort_key = [
                callable_distance(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, documents)
            ]
            reverse = False

        elif distance_measure_mode == "cosine":
            # TOOD: multiprocess this cos this is weak
            sort_key = [
                1 - spatial_distance.cosine(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, documents)
            ]
            reverse = True

        elif distance_measure_mode == "l2":
            sort_key = [
                spatial_distance.euclidean(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, documents)
            ]
            reverse = False

        else:
            raise ValueError("Need valid distance measure mode or callable distance")

        doc_utils.set_field_across_documents(score_field, sort_key, documents)

        return sorted(documents, reverse=reverse, key=lambda x: x[score_field])
