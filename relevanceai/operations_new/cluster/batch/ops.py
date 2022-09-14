"""
Batch Cluster Operations
"""
from typing import Optional

from relevanceai.dataset import Dataset
from relevanceai.operations_new.cluster.ops import ClusterOps
from relevanceai.operations_new.cluster.batch.transform import BatchClusterTransform
from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase

from relevanceai.constants.links import EXPLORER_APP_LINK
from relevanceai.operations_new.ops_manager import OperationManager
from relevanceai.operations_new.ops_run import PullTransformPush


class BatchClusterOps(BatchClusterTransform, ClusterOps):
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

    def fit(self, chunk):
        vectors = self.get_field_across_documents(
            self.vector_fields[0], chunk, missing_treatment="skip"
        )
        if hasattr(self.model, "n_clusters"):
            if len(vectors) < self.model.n_clusters:
                return chunk
        self.model.partial_fit(vectors)
        return chunk

    def run(
        self,
        dataset: Dataset,
        filters: list = None,
        chunksize: Optional[int] = None,
        **kwargs
    ):
        """
        Run batch clustering
        """
        from tqdm.auto import tqdm

        with OperationManager(dataset=dataset, operation=self) as dataset:
            tqdm.write("\nFitting Model...")
            ptp = PullTransformPush(
                dataset=dataset,
                func=self.fit,
                pull_chunksize=chunksize,
                push_chunksize=chunksize,
                filters=filters,
                select_fields=self.vector_fields,
                show_progress_bar=True,
                **kwargs
            )
            ptp.run()

            kwargs.pop("transform_chunksize")

            tqdm.write("\nPredicting Documents...")
            ptp = PullTransformPush(
                dataset=dataset,
                func=self.transform,
                pull_chunksize=chunksize,
                push_chunksize=chunksize,
                filters=filters,
                select_fields=self.vector_fields,
                show_progress_bar=True,
                **kwargs
            )
            ptp.run()

        tqdm.write("\nConfigure your new explore app below:")
        tqdm.write(EXPLORER_APP_LINK.format(dataset.dataset_id))
        return
