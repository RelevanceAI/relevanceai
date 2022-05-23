"""
OperationBase
"""
from relevanceai.operations_new.base import OperationBase


class BatchClusterBase(OperationBase):
    def partial_fit(self, documents):
        # Run partial fitting on documents
        pass

    def transform(self, documents):
        # Transform the documents
        pass
