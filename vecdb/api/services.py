"""Services class
"""
from vecdb.base import Base
from vecdb.api.aggregate import Aggregate
from vecdb.api.cluster import Cluster
from vecdb.api.encoders import Encoders
from vecdb.api.recommend import Recommend
from vecdb.api.search import Search
from vecdb.vis.projection import Projection


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
        self.projection = Projection(project=project, api_key=api_key, base_url=base_url)