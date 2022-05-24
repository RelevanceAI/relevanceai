"""
All functions related to running operations on datasets
"""
from datetime import datetime
from typing import Any, Dict, Optional

from relevanceai.dataset import Dataset
from relevanceai.operations_new.context import Upload
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
        with Upload(
            dataset=dataset,
            operation=self,
        ) as dataset:
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

    def run_in_chunks(
        self,
        dataset: Dataset,
        select_fields: list,
        filters: list,
        *args,
        **kwargs,
    ):
        """It takes a dataset, filters it, and then runs a transform function on each chunk of the filtered
        dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset,
        select_fields : list
            list of fields to select from the dataset
        filters : list
            list = []

        """
        with Upload(
            dataset=dataset,
            operation=self,
        ) as dataset:

            for chunk in dataset.chunk_dataset(
                select_fields=select_fields,
                filters=filters,
            ):
                updated_chunk = self.transform(
                    chunk,
                    *args,
                    **kwargs,
                )
                dataset.upsert_documents(updated_chunk)

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
            values = self.get_operation_metada()

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
