from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from relevanceai._api.endpoints.reports.clusters import ClusterReportClient


class ReportsClient(_Base):
    def __init__(self, credentials: Credentials):
        self.clusters = ClusterReportClient(credentials)
        super().__init__(credentials)
