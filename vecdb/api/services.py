"""Services class
"""
from ..base import Base
from .aggregate import Aggregate
from .cluster import Cluster
from .encoders import Encoders
from .recommend import Recommend
from .search import Search


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
