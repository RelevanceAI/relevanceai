"""
All functions related to running operations on datasets
"""
from datetime import datetime

from relevanceai.dataset import Dataset
from relevanceai.operations_new.base import OperationBase


class OperationRun(OperationBase):
    """
    All functions related to running operations on datasets
    """

    def run(
        self,
        dataset: Dataset,
        select_fields: list = None,
        filters: list = None,
        *args,
        **kwargs
    ):
        # A default run on all datasets
        documents = dataset.get_all_documents(
            select_fields=select_fields, filters=filters
        )
        # Loop through all documents
        updated_documents = self.transform(documents, *args, **kwargs)
        results = dataset.upsert_documents(updated_documents)
        # TODO: update values
        self.store_operation_metadata(
            dataset=dataset,
            values={
                "select_fields": select_fields,
                **kwargs,
            },
        )
        return results

    def run_in_chunks(
        self, dataset: Dataset, select_fields: list, filters: list, *args, **kwargs
    ):
        self.store_operation_metadata(dataset=dataset, values=kwargs)
        for chunk in dataset.chunk_dataset(
            select_fields=select_fields, filters=filters
        ):
            new_chunk = self.transform(chunk)
            dataset.upsert_documents(new_chunk)

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
                timestamp: {"operation": self.name, "parameters": str(values)}
            }
        }
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
