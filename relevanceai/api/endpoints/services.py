"""Services class
"""
from relevanceai.base import Base

from relevanceai.api.endpoints.encoders import Encoders
from relevanceai.api.endpoints.cluster import Cluster
from relevanceai.api.endpoints.search import Search
from relevanceai.api.endpoints.aggregate import Aggregate
from relevanceai.api.endpoints.recommend import Recommend
from relevanceai.api.endpoints.tagger import Tagger
from relevanceai.api.endpoints.prediction import Prediction
from relevanceai.api.endpoints.wordclouds import Wordclouds

class Services(Base):
    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.encoders = Encoders(project=project, api_key=api_key)
        self.cluster = Cluster(project=project, api_key=api_key)
        self.search = Search(project=project, api_key=api_key)
        self.aggregate = Aggregate(project=project, api_key=api_key)
        self.recommend = Recommend(project=project, api_key=api_key)
        self.tagger = Tagger(project=project, api_key=api_key)
        self.prediction = Prediction(project=project, api_key=api_key)
        self.wordclouds = Wordclouds(project=project, api_key=api_key)
        super().__init__(project, api_key)

    def document_diff(self, doc: dict, docs_to_compare: list, difference_fields: list = []):
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
            }
        )

