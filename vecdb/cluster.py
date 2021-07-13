from .base import Base
from .centroids import Centroids

class Cluster(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        self.centroids = Centroids(project=project, api_key=api_key, base_url=base_url)
