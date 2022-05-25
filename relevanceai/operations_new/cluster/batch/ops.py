"""
Batch Cluster Operations
"""
from relevanceai.operations_new.cluster.batch.base import BatchClusterBase
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.cluster.ops import ClusterOps
from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase
from relevanceai.dataset import Dataset
from typing import Any


class BatchClusterOps(BatchClusterBase, ClusterOps):
    """Batch Clustering related Operations"""

    def __init__(
        self,
        vector_fields,
        alias: str = None,
        model: BatchClusterModelBase = None,
        model_kwargs=None,
        dataset_id: str = None,
        cluster_field="_cluster_",
        verbose: bool = False,
        **kwargs
    ):
        if len(vector_fields) > 1:
            raise NotImplementedError(
                "Currently we do not support more than 1 vector field."
            )

        if dataset_id is not None:
            self.dataset_id = dataset_id
        self.vector_fields = vector_fields
        self.cluster_field = cluster_field
        self.verbose = verbose
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = str(model)

        if model_kwargs is None:
            model_kwargs = {}

        self.model_kwargs = model_kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        super().__init__(
            vector_fields=vector_fields,
            model="MiniBatchKmeans" if model is None else model,
            model_kwargs=model_kwargs,
            **kwargs
        )

        self.alias = self._get_alias(alias)

    def run(self, dataset: Dataset, filters: list = None):
        """
        Run batch clustering
        """
        # TODO:
        # Avoid looping through dataset twice
        print("Fitting...")
        for chunk in dataset.chunk_dataset(
            select_fields=self.vector_fields, filters=filters
        ):
            vectors = self.get_field_across_documents(
                self.vector_fields[0], chunk, missing_treatment="skip"
            )
            self.model.partial_fit(vectors)

        print("Predicting...")
        for chunk in dataset.chunk_dataset(select_fields=self.vector_fields):
            # Provide a chunk
            chunk = self.transform(chunk)
            results = dataset.upsert_documents(chunk)

        return
