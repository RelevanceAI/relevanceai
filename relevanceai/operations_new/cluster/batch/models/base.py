"""
The base for a partial cluster model base.
"""
from abc import abstractmethod, ABC
from relevanceai.operations_new.cluster.models.base import _ModelUtils


class BatchClusterModelBase(_ModelUtils, ABC):
    @abstractmethod
    def partial_fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError
