"""
All functions related to running operations on datasets
"""
from datetime import datetime
from typing import Any, Dict, Optional

from relevanceai.dataset import Dataset
from relevanceai.operations_new.base import OperationBase


class OperationRun(OperationBase):
    """
    All functions related to running operations on datasets
    """

    def run(
        self,
        dataset: Dataset,
        batched: Optional[bool] = False,
        chunksize: Optional[int] = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        *args,
        **kwargs,
    ):
        """It takes a dataset, and then it gets all the documents from that dataset. Then it transforms the
        documents and then it upserts the documents.

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        select_fields : list
            list = None,
        filters : list
            list = None,

        """

        from relevanceai.operations_new.manager import OperationManager

        with OperationManager(
            dataset=dataset,
            operation=self,
        ) as dataset:

            if batched:
                for chunk in dataset.chunk_dataset(
                    select_fields=select_fields,
                    filters=filters,
                    chunksize=chunksize,
                ):
                    updated_chunk = self.transform(
                        chunk,
                        *args,
                        **kwargs,
                    )
                    dataset.upsert_documents(updated_chunk)
            else:
                documents = dataset.get_all_documents(
                    select_fields=select_fields,
                    filters=filters,
                )
                updated_documents = self.transform(
                    documents,
                    *args,
                    **kwargs,
                )
                dataset.upsert_documents(updated_documents)

    def store_operation_metadata(
        self,
        dataset: Dataset,
        values: Optional[Dict[str, Any]] = None,
    ):
        """This function stores metadata about operators

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        values : Optional[Dict[str, Any]]
            Optional[Dict[str, Any]] = None,

        Returns
        -------
            The dataset object with the metadata appended to it.

        .. code-block::

            {
                "_operationhistory_": {
                    "1-1-1-17-2-3": {
                        "operation": "vector", "model_name": "miniLm"
                    },
                }
            }

        """
        if values is None:
            values = self.get_operation_metadata()

        print("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = {
            "_operationhistory_": {
                timestamp: {
                    "operation": self.name,
                    "parameters": str(values),
                }
            }
        }
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
