"""Services class
"""
from typing import Optional

from relevanceai.utils.base import _Base

from relevanceai._api.endpoints.services.cluster import ClusterClient
from relevanceai._api.endpoints.services.search import SearchClient
from relevanceai._api.endpoints.services.aggregate import AggregateClient
from relevanceai._api.endpoints.services.recommend import RecommendClient
from relevanceai._api.endpoints.services.tagger import TaggerClient
from relevanceai._api.endpoints.services.prediction import PredictionClient
from relevanceai._api.endpoints.datasets.cluster.centroids import CentroidsClient


class ServicesClient(_Base):
    def __init__(self, credentials):
        self.cluster = ClusterClient(credentials)
        self.search = SearchClient(credentials)
        self.aggregate = AggregateClient(credentials)
        self.recommend = RecommendClient(credentials)
        self.tagger = TaggerClient(credentials)
        self.prediction = PredictionClient(credentials)
        self.centroids = CentroidsClient(credentials)
        super().__init__(credentials)

    def document_diff(
        self,
        doc: dict,
        documents_to_compare: list,
        difference_fields: Optional[list] = None,
    ):
        """
        Find differences between documents

        Parameters
        ----------
        doc: dict
            Main document to compare other documents against.
        documents_to_compare: list
            Other documents to compare against the main document.
        difference_fields: list
            Fields to compare. Defaults to [], which compares all fields.

        """
        difference_fields = [] if difference_fields is None else difference_fields

        return self.make_http_request(
            endpoint=f"/services/document_diff",
            method="POST",
            parameters={
                "doc": doc,
                "documents_to_compare": documents_to_compare,
                "difference_fields": difference_fields,
            },
        )
