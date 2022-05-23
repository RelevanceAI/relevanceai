from relevanceai.dataset import Dataset
from relevanceai._api import APIClient

from relevanceai.operations_new.run import OperationRun


class OperationsAPILogger:
    pass


class OperationAPIBase(APIClient, OperationRun, OperationsAPILogger):
    def run_on(self, dataset: Dataset, **kwargs):
        for documents in dataset.chunk_dataset():
            documents = self.run(documents=documents, **kwargs)
            dataset.upsert_documents(documents)

        dataset.store_operation_metadata(
            operation=self.name,
            values=str(kwargs),
        )
        return

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
