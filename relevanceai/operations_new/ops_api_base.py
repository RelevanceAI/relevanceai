from relevanceai.dataset import Dataset
from relevanceai._api import APIClient

from relevanceai.operations_new.ops_run_base import OperationRunBase


class OperationsAPILogger:
    pass


class OperationAPIBase(APIClient, OperationRunBase, OperationsAPILogger):
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
