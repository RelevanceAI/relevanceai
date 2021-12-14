import scipy.spatial.distance as spatial_distance
from relevanceai.base import _Base
from doc_utils.doc_utils import DocUtils
from relevanceai.vector_tools.constants import NEAREST_NEIGHBOURS

doc_utils = DocUtils()


class NearestNeighbours(_Base, DocUtils):
    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    @staticmethod
    def get_nearest_neighbours(
        docs: list,
        vector: list,
        vector_field: str,
        distance_measure_mode: NEAREST_NEIGHBOURS = "cosine",
        callable_distance=None,
    ):

        if callable_distance:
            sort_key = [
                callable_distance(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, docs)
            ]
            reverse = False

        elif distance_measure_mode == "cosine":
            sort_key = [
                1 - spatial_distance.cosine(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, docs)
            ]
            reverse = True

        elif distance_measure_mode == "l2":
            sort_key = [
                spatial_distance.euclidean(i, vector)
                for i in doc_utils.get_field_across_documents(vector_field, docs)
            ]
            reverse = False

        else:
            raise ValueError("Need valid distance measure mode or callable distance")

        doc_utils.set_field_across_documents(
            "nearest_neighbour_distance", sort_key, docs
        )

        return sorted(
            docs, reverse=reverse, key=lambda x: x["nearest_neighbour_distance"]
        )
