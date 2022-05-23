"""
Base Operations Class
"""
from abc import ABC, abstractmethod

from typing import Any, Dict, List
from datetime import datetime

from relevanceai.utils import DocUtils
from relevanceai.dataset import Dataset


class OperationBase(ABC, DocUtils):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.run(*args, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def run(
        self, dataset: Dataset, select_fields: list, filters: list, *args, **kwargs
    ):
        # Run on all datasets
        documents = dataset.get_all_documents(
            select_fields=select_fields, filters=filters
        )
        # Loop through all documents
        updated_documents = self.transform(documents, *args, **kwargs)
        return dataset.upsert_documents(updated_documents)

    def run_in_chunks(self, dataset: Dataset, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def transform(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """The function is an abstract method that raises a NotImplementedError if it is not implemented"""
        raise NotImplementedError

    def _check_vector_fields(self):
        # check the vector fields
        if hasattr(self, "vector_fields"):
            for vector_field in self.vector_fields:
                if not vector_field.endswith("_vector_"):
                    raise ValueError(
                        "Invalid vector field. Ensure they end in `_vector_`."
                    )
        # TODO: check the schema
        if hasattr(self, "dataset"):
            for vector_field in self.vector_fields:
                if hasattr(self.dataset, "schema"):
                    assert vector_field in self.dataset.schema

    def _check_alias(self):
        if self.alias is None:
            raise ValueError("alias is not set. Please supply alias=")

    def store_operation_metadata(self, dataset: Dataset, operation: str, values: str):
        """
        Store metadata about operators
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
                timestamp: {"operation": operation, "parameters": values}
            }
        }
        # Gets metadata and appends to the operation history
        return dataset.upsert_metadata(metadata)
