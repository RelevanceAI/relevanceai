from typing import Union, Any, Optional, List

from doc_utils import DocUtils

from relevanceai.client.helpers import Credentials
from relevanceai.operations.dr.dim_reduction import PCA
from relevanceai.operations.dr.dim_reduction import TSNE
from relevanceai.operations.dr.dim_reduction import Ivis
from relevanceai.operations.dr.dim_reduction import UMAP
from relevanceai.operations import BaseOps
from relevanceai._api import APIClient


class ReduceDimensionsOps(APIClient, BaseOps):
    def __init__(
        self,
        credentials: Credentials,
        n_components: int,
        model: Union[PCA, TSNE, Ivis, PCA, str, Any],
        dr_field: str = "_dr_",
        verbose: bool = True,
        dataset_id: Optional[str] = None,
        vector_fields: Optional[List] = None,
        alias: Optional[str] = None,
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
        self.dataset_id = dataset_id
        self.vector_fields = vector_fields
        self.alias = alias

        super().__init__(credentials)

    def fit(
        self,
        dataset_id: Optional[str] = None,
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
    ):
        """
        Reduce Dimensions

        Example
        ---------

        dataset_id: Optional[str]
            The dataset to run dimensionality reduction on
        vector_fields: Optional[List[str]]
            List of vector fields
        alias: Optional[str]
            Alias to store dimensionality reduction model

        Parameters
        -------------

        fields: list
            The list of fields to run dimensionality reduction on. Currently
            only supports 1 field.
        documents: list
            The list of documents to run dimensionality reduction on
        inplace: bool
            If True, replaces the original documents, otherwise it returns
            a new set of documents with only the dr vectors in it and the _id


        """
        if dataset_id is None:
            dataset_id = self.dataset_id
        if vector_fields is None:
            vector_fields = self.vector_fields
        if alias is None:
            alias = self.alias

        documents = self._get_all_documents(
            dataset_id, select_fields=vector_fields, include_vector=True
        )

        dr_documents = self.model.fit_transform_documents(
            vector_field=vector_fields[0],  # type: ignore
            documents=documents,
            alias=alias,  # type: ignore
            dims=self.n_components,
        )

        return self._update_documents(dataset_id=dataset_id, documents=dr_documents)  # type: ignore

    def operate(
        self,
        dataset_id: Optional[str] = None,
        vector_fields: Optional[List[str]] = None,
        alias: Optional[str] = None,
    ):
        """Operate the dashboard"""
        return self.fit(dataset_id=dataset_id, vector_fields=vector_fields, alias=alias)
