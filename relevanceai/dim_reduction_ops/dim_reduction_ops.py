"""
Dimensionality Reduction Ops

.. warning::
    This is a beta feature and is currently in development.

Reducing dimensions for just documents.

.. code-block::

    from relevanceai import Client 
    client = Client()

    from relevanceai.datasets import mock_documents
    docs = mock_documents(10)

    from relevanceai.dim_reduction_ops import ReduceDimensionsOps
    from sklearn.decomposition import PCA
    model = PCA(n_components=2)
    dim_reducer = ReduceDimensionsOps(model)
    dim_reducer.reduce_dimensions(fields=["sample_1_vector_"], documents=documents)

"""
import copy
from doc_utils import DocUtils


class ReduceDimensionOps(DocUtils):
    def __init__(self, model, alias: str):
        self.model = model
        self.alias = alias

    def reduce_dimensions(self, fields: list, documents: list, inplace: bool = True):
        """
        Reduce Dimensions
        """
        self.fields = fields
        self.documents = documents
        values = self.get_fields_across_documents(fields, documents)
        dr_values = self.model.fit_transform(values)
        if inplace:
            self.set_dr_field_across_documents(fields, dr_values, documents)
            return documents
        else:
            new_documents = copy.deepcopy(documents)
            self.set_dr_field_across_documents(fields, dr_values, new_documents)
            return new_documents

    def set_dr_field_across_documents(self, fields: list, values, documents):
        vector_fields_joined = ".".join(fields)
        field_name = f"_dr_.{vector_fields_joined}.{self.alias}"
        self.set_field_across_documents(field_name, values, documents)
        return documents
