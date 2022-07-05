"""
Base class for base.py to inherit.
All functions related to running operations on datasets.
"""
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from relevanceai.dataset import Dataset
from relevanceai.operations_new.transform_base import TransformBase

from relevanceai.utils import fire_and_forget


class OperationRun(TransformBase):
    """
    All functions related to running transforms as an operation on datasets
    """

    def is_chunk_valid(self, chunk):
        return chunk is not None and len(chunk) > 0

    def run(
        self,
        dataset: Dataset,
        batched: Optional[bool] = False,
        chunksize: Optional[int] = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        output_fields: Optional[list] = None,
        refresh: bool = False,
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
            Used to determine which fields to retrieve for filters
        output_fields: list
            Used to determine which output fields are missing to continue running operation

        filters : list
            list = None,

        """

        if filters is None:
            filters = []
        if select_fields is None:
            select_fields = []

        # store this
        if hasattr(dataset, "dataset_id"):
            self.dataset_id = dataset.dataset_id

        schema = dataset.schema

        self._check_fields_in_schema(select_fields)

        filters += [
            {
                "filter_type": "or",
                "condition_value": [
                    {
                        "field": field,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": " ",
                    }
                    for field in select_fields
                ],
            }
        ]

        # add a checkmark for output fields
        if not refresh and output_fields is not None and len(output_fields) > 0:
            filters += [
                {
                    "field": output_fields[0],
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": " ",
                }
            ]

        # needs to be here due to circular imports
        from relevanceai.operations_new.ops_manager import OperationManager

        with OperationManager(
            dataset=dataset,
            operation=self,
        ) as dataset:

            if batched:
                self.batch_transform_upsert(
                    dataset=dataset,
                    select_fields=select_fields,
                    filters=filters,
                    chunksize=chunksize,
                    **kwargs,
                )
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

    def batch_transform_upsert(
        self,
        dataset: Dataset,
        select_fields: list = None,
        filters: list = None,
        chunksize: int = None,
        max_active_threads: int = 2,
        timeout: int = 30,
        *args,
        **kwargs,
    ):
        # Here we limit the number of threadsA
        thread_count = 0

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
            if self.is_chunk_valid(updated_chunk):

                @fire_and_forget
                def fire_upsert_docs():
                    dataset.upsert_documents(updated_chunk)

                # Add a check for timecount
                thread_count += 1
                if thread_count >= max_active_threads:
                    # Check if thread count decreases
                    checker = 0
                    curr_thread_count = threading.active_count()
                    while (
                        threading.active_count() >= curr_thread_count
                        and checker < timeout
                    ):
                        time.sleep(1)
                        checker += 1
                    thread_count -= 1

                fire_upsert_docs()

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
                        "operation": "vector",
                        "model_name": "miniLm"
                    },
                }
            }

        """
        if values is None:
            values = self.get_operation_metadata()

        print("Storing operation metadata...")
        timestamp = str(datetime.now().timestamp()).replace(".", "-")
        metadata = dataset.metadata.to_dict()
        if "_operationhistory_" not in metadata:
            metadata["_operationhistory_"] = {}
        metadata["_operationhistory_"].update(
            {
                timestamp: {
                    "operation": self.name,
                    "parameters": str(values),
                }
            }
        )
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
