"""Services class
"""
from relevanceai.base import _Base

from relevanceai.api.endpoints.encoders import EncodersClient
from relevanceai.api.endpoints.cluster import ClusterClient
from relevanceai.api.endpoints.search import SearchClient
from relevanceai.api.endpoints.aggregate import AggregateClient
from relevanceai.api.endpoints.recommend import RecommendClient
from relevanceai.api.endpoints.tagger import TaggerClient
from relevanceai.api.endpoints.prediction import PredictionClient
from relevanceai.api.endpoints.wordclouds import WordcloudsClient


class ServicesClient(_Base):
    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.encoders = EncodersClient(project=project, api_key=api_key)
        self.cluster = ClusterClient(project=project, api_key=api_key)
        self.search = SearchClient(project=project, api_key=api_key)
        self.aggregate = AggregateClient(project=project, api_key=api_key)
        self.recommend = RecommendClient(project=project, api_key=api_key)
        self.tagger = TaggerClient(project=project, api_key=api_key)
        self.prediction = PredictionClient(project=project, api_key=api_key)
        self.wordclouds = WordcloudsClient(project=project, api_key=api_key)
        super().__init__(project, api_key)

    def document_diff(
        self, doc: dict, docs_to_compare: list, difference_fields: list = []
    ):
        """
        Find differences between documents

        Parameters
        ----------
        doc: dict
            Main document to compare other documents against.
        docs_to_compare: list
            Other documents to compare against the main document.
        difference_fields: list
            Fields to compare. Defaults to [], which compares all fields.

        """
        return self.make_http_request(
            endpoint=f"/services/document_diff",
            method="POST",
            parameters={
                "doc": doc,
                "docs_to_compare": docs_to_compare,
                "difference_fields": difference_fields,
            },
        )
