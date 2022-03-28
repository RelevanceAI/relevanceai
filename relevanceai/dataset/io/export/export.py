from typing import Dict, Any, Callable

from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.decorators.version import added
from relevanceai.dataset.io.export.csv import CSVExport
from relevanceai.dataset.io.export.dict import DictExport
from relevanceai.dataset.io.export.pandas import PandasExport


class Export(CSVExport, DictExport, PandasExport):
    """Exports"""

    @track
    @added(version="2.0.0")
    def to_dataset(
        self,
        child_dataset_id: str,
        filters: list = None,
        filter_condition: Callable = None,
        chunksize: int = 20,
    ):
        """Export this current dataset to another dataset"""
        for i, chunk in enumerate(
            self.chunk_dataset(chunksize=chunksize, filters=filters)
        ):
            if filter_condition is not None:
                chunk = [d for d in chunk if filter_condition(d)]
            if i == 0:
                self._insert_documents(child_dataset_id, documents=chunk, verbose=True)
            else:
                self._insert_documents(child_dataset_id, documents=chunk, verbose=False)
        self.datasets.post_metadata(
            dataset_id=child_dataset_id, metadata={"parent_dataset_id": self.dataset_id}
        )
        # Useful for viewing flowcharts
        metadata: Dict[str, Any] = {}
        metadata["child_dataset_ids"] = []
        metadata["child_dataset_ids"].append(child_dataset_id)
        self.upsert_metadata(metadata=metadata)
