from relevanceai.dataset import Dataset
from relevanceai._api import APIClient

from relevanceai.operations_new.run import OperationRun


class OperationsAPILogger:
    pass


class OperationAPIBase(APIClient, OperationRun, OperationsAPILogger):
    @classmethod
    def from_credentials(self, *args, **kwargs):
        """
        .. code-block::

            ClusterOps.from_dataset()
        """
        raise NotImplementedError

    @classmethod
    def from_dataset(self, *args, **kwargs):
        raise NotImplementedError
