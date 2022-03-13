"""Services class
"""
from typing import Optional

from relevanceai.package_utils.base import _Base

from relevanceai.api.endpoints.services.encoders import EncodersClient
from relevanceai.api.endpoints.services.cluster import ClusterClient
from relevanceai.api.endpoints.services.search import SearchClient
from relevanceai.api.endpoints.services.aggregate import AggregateClient
from relevanceai.api.endpoints.services.recommend import RecommendClient
from relevanceai.api.endpoints.services.tagger import TaggerClient
from relevanceai.api.endpoints.services.prediction import PredictionClient
from relevanceai.api.endpoints.services.wordclouds import WordcloudsClient


class ServicesClient(_Base):
    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

        self.encoders = EncodersClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.cluster = ClusterClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.search = SearchClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.aggregate = AggregateClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.recommend = RecommendClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.tagger = TaggerClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.prediction = PredictionClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        self.wordclouds = WordcloudsClient(
            project=project, api_key=api_key, firebase_uid=firebase_uid
        )
        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

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
