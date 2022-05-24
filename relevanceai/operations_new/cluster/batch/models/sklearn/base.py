from relevanceai.operations_new.cluster.batch.models.base import BatchClusterModelBase


class SklearnBatchClusterBase(BatchClusterModelBase):
    model: BatchClusterModelBase

    def partial_fit(self, X):
        return self.model.partial_fit(X)

    def predict(self, X):
        return self.model.predict(X)
