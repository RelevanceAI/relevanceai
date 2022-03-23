from relevanceai.workflows.cluster import ClusterOps
from relevanceai.workflows.dr import ReduceDimensionsOps


class Workflows(ClusterOps, ReduceDimensionsOps):
    def __init__(self):
        super().__init__()
