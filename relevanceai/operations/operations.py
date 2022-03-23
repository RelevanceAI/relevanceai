from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.dr import ReduceDimensionsOps


class Operations(ClusterOps, ReduceDimensionsOps):
    def __init__(self):
        super().__init__()
