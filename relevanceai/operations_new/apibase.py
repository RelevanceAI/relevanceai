from relevanceai.dataset import Dataset
from relevanceai._api import APIClient

from relevanceai.operations_new.base import OperationBase


class OperationsAPILogger:
    pass


class OperationsAPIBase(APIClient, OperationBase, OperationsAPILogger):
    def run_for(self, dataset: Dataset, **kwargs):
        for documents in dataset.chunk_dataset():
            documents = self.run(documents=documents, **kwargs)
            dataset.upsert_documents(documents)

        dataset.store_operation_metadata(
            operation=self.name,
            values=str(kwargs),
        )
        return
