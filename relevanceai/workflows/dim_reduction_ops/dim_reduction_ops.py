"""
Dimensionality Reduction Ops

.. warning::
    This is a beta feature and is currently in development.

Reducing dimensions for just documents.

.. code-block::

    from relevanceai import Client 
    client = Client()

    from relevanceai.package_utils.datasets import mock_documents
    docs = mock_documents(10)

    from relevanceai.dim_reduction_ops import ReduceDimensionsOps
    from sklearn.decomposition import PCA
    model = PCA(n_components=2)
    dim_reducer = ReduceDimensionsOps(model)
    dim_reducer.reduce_dimensions(fields=["sample_1_vector_"], documents=documents)

"""
import copy
from doc_utils import DocUtils


class ReduceDimensionsOps(DocUtils):
    def __init__(self, model, alias: str):
        """
        Dim Reduction Ops

        Parameters
        --------------

        model
            The model to run dimensionality reduction. This requires a `fit_transform` method.
        alias: str
            The alias of the dimensionality reduction
        """
        self.model = model
        if not hasattr(model, "fit_transform"):
            raise AttributeError("‼️ Model needs to have a fit_transform method.")
        self.alias = alias

    def reduce_dimensions(self, fields: list, documents: list, inplace: bool = True):
        """
        Reduce Dimensions

        Parameters
        --------------

        fields: list
            The list of fields to run dimensionality reduction on. Currently
            only supports 1 field.
        documents: list
            The list of documents to run dimensionality reduction on
        inplace: bool
            If True, replaces the original documents, otherwise it returns
            a new set of documents with only the dr vectors in it and the _id
        """
        self.fields = fields
        self.documents = documents
        if len(fields) == 1:
            values = self.get_field_across_documents(fields[0], documents)
        else:
            raise ValueError("Supporting multiple fields not supported yet.")
        dr_values = self.model.fit_transform(values)
        new_documents = self.set_dr_field_across_documents(
            fields, dr_values, documents, inplace=inplace
        )
        return new_documents

    def set_dr_field_across_documents(
        self, fields: list, values: list, documents: list, inplace: bool = True
    ):
        """
        Setting the DR field allows users to quickly follow Relevance AI
        best practice in naming.
        It follows `_dr_.*field_name*.alias`

        Parameters
        -------------

        fields:
            The list of fields to set
        values:
            The list of values to set
        documents:
            List of documents to set

        """
        fields_joined = ".".join(fields)
        field_name = f"_dr_.{fields_joined}.{self.alias}"

        if inplace:
            self.set_field_across_documents(field_name, values, documents)
            return documents
        new_documents = [{"_id": d["_id"]} for d in documents]
        self.set_field_across_documents(field_name, values, new_documents)
        return new_documents
