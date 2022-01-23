from abc import ABC
from abc import abstractmethod

from relevanceai.config import CONFIG
from relevanceai.transport import Transport
from relevanceai.logger import LoguruLogger


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger):
    """Base class for all relevanceai client utilities"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.config = CONFIG
        # Initialize logger
        super().__init__()


class ClusterBase(ABC):
    def __init__(self, clusterer):
        self.clusterer = clusterer

    @abstractmethod
    def fit_transform(self, vectors):
        raise NotImplementedError

    @abstractmethod
    def get_centroids(self):
        raise NotImplementedError

    def metadata(self):
        if hasattr(self, __dict__):
            return self.clusterer.__dict__

        elif hasattr(self, __name__):
            return self.__name__

        else:
            return "Clusterer"
