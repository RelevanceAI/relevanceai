from relevanceai.base import _Base
from relevanceai.api.endpoints.datasets.cluster.centroids import CentroidsClient


class ClusterClient(_Base):
    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.centroids = CentroidsClient(project=project, api_key=api_key)
        super().__init__(project=project, api_key=api_key)
