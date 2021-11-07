"""Services class
"""
from relevanceai.base import Base

from relevanceai.api.encoders import Encoders
from relevanceai.api.cluster import Cluster
from relevanceai.api.search import Search
from relevanceai.api.aggregate import Aggregate
from relevanceai.api.recommend import Recommend

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
        super().__init__(project, api_key, base_url)
