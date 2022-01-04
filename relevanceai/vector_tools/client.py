from relevanceai.base import _Base
from relevanceai.vector_tools.cluster import Cluster
from relevanceai.vector_tools.dim_reduction import DimReduction
from relevanceai.vector_tools.nearest_neighbours import NearestNeighbours


class VectorTools(_Base):
    """Vector Tools Client"""

    def __init__(self, project: str, api_key: str):
        self.cluster = Cluster(project=project, api_key=api_key)
        self.dim_reduction = DimReduction(project=project, api_key=api_key)
        self.nearest_neighbours = NearestNeighbours(project=project, api_key=api_key)
        super().__init__(project, api_key)
