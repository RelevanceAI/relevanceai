from typing import Union, List, Any

from doc_utils import DocUtils

from relevanceai.client.helpers import Credentials
from relevanceai.operations.dr.dim_reduction import PCA
from relevanceai.operations.dr.dim_reduction import TSNE
from relevanceai.operations.dr.dim_reduction import Ivis
from relevanceai.operations.dr.dim_reduction import UMAP
from relevanceai._api import APIClient


class ReduceDimensionsOps(APIClient, DocUtils):
    def __init__(
        self,
        credentials: Credentials,
        n_components: int,
        model: Union[PCA, TSNE, Ivis, PCA, str, Any],
        dr_field: str = "_dr_",
        verbose: bool = True,
    ):
        if isinstance(model, str):
            algorithm = model.upper()
            if algorithm == "PCA":

                model = PCA()
            elif algorithm == "TSNE":

                model = TSNE()
            elif algorithm == "UMAP":

                model = UMAP()
            elif algorithm == "IVIS":

                model = Ivis()
            else:
                raise ValueError()

        self.model = model
        self.dr_field = dr_field
        self.verbose = verbose
        self.n_components = n_components

        super().__init__(credentials)

    def fit(
        self,
        dataset_id: str,
        vector_fields: str,
        alias: str,
    ):
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
        documents = self._get_all_documents(
            dataset_id, select_fields=vector_fields, include_vector=True
        )

        dr_documents = self.model.fit_transform_documents(
            vector_field=vector_fields[0],
            documents=documents,
            alias=alias,
            dims=self.n_components,
        )

        return self.update_documents(dataset_id, dr_documents)
