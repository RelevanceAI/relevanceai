"""
All functions related to running operations on datasets
"""

from relevanceai.dataset import Dataset
from relevanceai.operations_new.base import OperationBase
from datetime import datetime


class OperationRun(OperationBase):
    """
    All functions related to running operations on datasets
    """

    def run(
        self, dataset: Dataset, select_fields: list, filters: list, *args, **kwargs
    ):
        # Run on all datasets
        documents = dataset.get_all_documents(
            select_fields=select_fields, filters=filters
        )
        # Loop through all documents
        updated_documents = self.transform(documents, *args, **kwargs)
        results = dataset.upsert_documents(updated_documents)
        # TODO: update values
        self.store_operation_metadata(
            dataset=dataset, values={"select_fields": select_fields, "dataset": dataset}
        )
        return results

    def run_in_chunks(
        self, dataset: Dataset, select_fields: list, filters: list, *args, **kwargs
    ):
        for i, chunk in enumerate(
            dataset.chunk_dataset(select_fields=select_fields, filters=filters)
        ):
            new_chunk = self.transform(chunk)
            dataset.upsert_documents(new_chunk)
            if i == 0:
                # We only need to store on the first operation
                self.store_operation_metadata(dataset=dataset, values=kwargs)

    def store_operation_metadata(self, dataset: Dataset, values: dict):
        """
        Store metadata about operators
        This is stored in the format:
        .. code-block::

            {
                "_operationhistory_": {
                    "1-1-1-17-2-3": {
                        "operation": "vector", "model_name": "miniLm"
                    },
                }
            }

        """
        print("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = {
            "_operationhistory_": {
                timestamp: {"operation": self.name, "parameters": values}
            }
        }
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
