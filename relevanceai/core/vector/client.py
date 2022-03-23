from relevanceai.utils.base import _Base
from relevanceai.core.dr.dim_reduction import DimReduction
from relevanceai.core.vector.local_nearest_neighbours import NearestNeighbours


class VectorTools(_Base):
    """Vector Tools Client"""

    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.dim_reduction = DimReduction(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.nearest_neighbours = NearestNeighbours(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)
