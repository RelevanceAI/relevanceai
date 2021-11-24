"""Services class
"""
from relevanceai.base import Base

from relevanceai.api.encoders import Encoders
from relevanceai.api.cluster import Cluster
from relevanceai.api.search import Search
from relevanceai.api.aggregate import Aggregate
from relevanceai.api.recommend import Recommend
from relevanceai.api.tagger import Tagger
from relevanceai.api.prediction import Prediction
from relevanceai.api.wordclouds import Wordclouds

class Services(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
        self.encoders = Encoders(project=project, api_key=api_key, base_url=base_url)
        self.cluster = Cluster(project=project, api_key=api_key, base_url=base_url)
        self.search = Search(project=project, api_key=api_key, base_url=base_url)
        self.aggregate = Aggregate(project=project, api_key=api_key, base_url=base_url)
        self.recommend = Recommend(project=project, api_key=api_key, base_url=base_url)
        self.tagger = Tagger(project=project, api_key=api_key, base_url=base_url)
        self.prediction = Prediction(project=project, api_key=api_key, base_url=base_url)
        self.wordclouds = Wordclouds(project=project, api_key=api_key, base_url=base_url)
        super().__init__(project, api_key, base_url)

    def document_diff(self, doc: dict, docs_to_compare: list, difference_fields: list = [], output_format: str = "json", verbose: bool = True):
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
            output_format=output_format,
            verbose=verbose,
        )

