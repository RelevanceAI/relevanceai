from relevanceai.operations_new.cluster.batch.models.sklearn.base import (
    SklearnBatchClusterBase,
)


class MiniBatchKMeans(SklearnBatchClusterBase):
    @property
    def name(self):
        return "minibatchkmeans"
