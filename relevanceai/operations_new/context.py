from typing import List, Dict, Any

from relevanceai.dataset import Dataset
from relevanceai.operations_new.run import OperationRun


class Upload:
    def __init__(
        self,
        dataset: Dataset,
        operation: OperationRun,
        metadata: Dict[str, Any],
    ):
        self.dataset = dataset
        self.operation = operation
        self.metadata = metadata

    def __enter__(self):
        return self.dataset

    def __exit__(self, *args, **kwargs):
        self.operation.store_operation_metadata(
            dataset=self.dataset,
            values=self.metadata,
        )

    @staticmethod
    def clean(
        before_docs: List[Dict[str, Any]],
        after_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        updated_documents = [
            {
                key: value
                for key, value in after_doc.items()
                if key not in before_doc or key == "_id"
            }
            for (before_doc, after_doc,) in zip(
                before_docs,
                after_docs,
            )
        ]
        return updated_documents
