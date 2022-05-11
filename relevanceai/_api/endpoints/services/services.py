"""Services class
"""
from typing import Optional

from relevanceai.utils.base import _Base

from relevanceai._api.endpoints.services.encoders import EncodersClient
from relevanceai._api.endpoints.services.cluster import ClusterClient
from relevanceai._api.endpoints.services.aggregate import AggregateClient
from relevanceai._api.endpoints.services.tagger import TaggerClient
from relevanceai._api.endpoints.services.wordclouds import WordcloudsClient
from relevanceai._api.endpoints.services.centroids import CentroidsClient


class ServicesClient(_Base):
    def __init__(self, credentials):
        self.encoders = EncodersClient(credentials)
        self.cluster = ClusterClient(credentials)
        self.aggregate = AggregateClient(credentials)
        self.tagger = TaggerClient(credentials)
        self.wordclouds = WordcloudsClient(credentials)
        self.centroids = CentroidsClient(credentials)
        super().__init__(credentials)
