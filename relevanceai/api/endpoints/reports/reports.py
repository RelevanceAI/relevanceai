from relevanceai.package_utils.base import _Base
from relevanceai.api.endpoints.reports.clusters import ClusterReportClient


class ReportsClient(_Base):
    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid
        self.clusters = ClusterReportClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)
