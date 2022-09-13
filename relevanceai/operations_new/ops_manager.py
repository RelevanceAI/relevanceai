from typing import List, Dict, Any, Optional

from relevanceai.dataset import Dataset
from relevanceai.operations_new.ops_run import OperationRun

from tqdm.auto import tqdm


class OperationManager:
    """
    Operation manager manages an operation.
    It handles storing operation manager.

    In future, it will also handle:
    - logging
    - timing
    """

    def __init__(
        self,
        dataset: Dataset,
        operation: OperationRun,
        metadata: Optional[Dict[str, Any]] = None,
        post_hooks: Optional[list] = None,
    ):
        self.dataset = dataset
        self.operation = operation
        self.metadata = metadata
        self.post_hooks = post_hooks if post_hooks is not None else []

    def __enter__(self):
        return self.dataset

    def __exit__(self, *args, **kwargs):
        from relevanceai.operations_new.cluster.ops import ClusterOps
        from relevanceai.operations_new.cluster.sub.ops import SubClusterOps
        from relevanceai.operations_new.cluster.batch.ops import BatchClusterOps

        if isinstance(self.operation, SubClusterOps):
            # Get the centroids if there are any
            # Once they are retrieved, we want to get the centroids of the new subcluster operation
            # Then we want to update the original centroids
            # Then upsert them back into the centroids dictionary
            try:
                from relevanceai.utils import suppress_stdout_stderr

                with suppress_stdout_stderr():
                    centroids = self.operation.centroids
                centroid_documents = {
                    d["_id"]: d[self.operation.vector_field] for d in centroids
                }
            except:
                centroid_documents = {}
            subcluster_centroid_documents = self.operation.get_centroid_documents()
            subcluster_centroid_documents = {
                d["_id"]: d[self.operation.vector_fields[0]]
                for d in subcluster_centroid_documents
            }
            centroid_documents.update(subcluster_centroid_documents)
            centroid_documents = [
                {"_id": k, self.operation.vector_fields[0]: v}
                for k, v in centroid_documents.items()
            ]

        elif isinstance(self.operation, ClusterOps):
            centroid_documents = self.operation.get_centroid_documents()

        elif isinstance(self.operation, BatchClusterOps):
            centroid_documents = self.operation.get_centroid_documents()

        if "centroid_documents" in locals():
            tqdm.write("Inserting centroids...")
            res = self.operation.insert_centroids(centroid_documents)
            tqdm.write("Centroids Intserted!")

        for h in self.post_hooks:
            h()

        self._store_default_parent_child()

        self.operation.store_operation_metadata(
            dataset=self.dataset,
            values=self.metadata,
        )

    def _store_default_parent_child(self):
        """We temporarily store the default parent child relationship
        where possible
        """
        try:
            if hasattr(self.operation, "select_fields"):
                # Based on the
                for i, field in enumerate(self.operation.select_fields):
                    self.dataset.update_field_children(
                        field=field,
                        field_children=[self.operation.output_fields[i]],
                        category=self.operation.name,  # Should this be the workflow ID
                        metadata={},
                    )
            elif hasattr(self.operation, "fields"):
                # Based on the
                for i, field in enumerate(self.operation.fields):
                    self.dataset.update_field_children(
                        field=field,
                        field_children=[self.operation.output_fields[i]],
                        category=self.operation.name,  # Should this be the workflow ID
                        metadata={},
                    )
            elif hasattr(self.operation, "text_fields"):
                # Based on the
                for i, field in enumerate(self.operation.text_fields):
                    self.dataset.update_field_children(
                        field=field,
                        field_children=[self.operation.output_fields[i]],
                        category=self.operation.name,  # Should this be the workflow ID
                        metadata={},
                    )
        except Exception as e:
            # TODO: rigorously test this with different operations
            # reason: `output_fields` are all calculated differently in
            # different spots for different operations
            print(e)

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
