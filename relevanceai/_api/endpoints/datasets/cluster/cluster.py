from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from relevanceai._api.endpoints.datasets.cluster.centroids import CentroidsClient


class ClusterClient(_Base):
    def __init__(self, credentials: Credentials):
        self.centroids = CentroidsClient(credentials)
        super().__init__(credentials)
