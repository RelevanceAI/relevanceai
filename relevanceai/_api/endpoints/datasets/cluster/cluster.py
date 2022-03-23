from relevanceai.utils.base import _Base
from relevanceai._api.endpoints.datasets.cluster.centroids import CentroidsClient


class ClusterClient(_Base):
    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

        self.centroids = CentroidsClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)
