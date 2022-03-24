from typing import Union, List, Any

from doc_utils import DocUtils

from relevanceai._api.client import BatchAPIClient

from relevanceai.operations.dr.dim_reduction import PCA
from relevanceai.operations.dr.dim_reduction import TSNE
from relevanceai.operations.dr.dim_reduction import Ivis
from relevanceai.operations.dr.dim_reduction import UMAP


class ReduceDimensionsOps(BatchAPIClient, DocUtils):
    def __init__(
        self,
        alias: str,
        project: str,
        api_key: str,
        firebase_uid: str,
        dataset_id: str,
        n_components: int,
        vector_fields: List[str],
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

        self.dr_field = dr_field
        self.verbose = verbose
        self.dataset_id = dataset_id
        self.model = model
        self.alias = alias
        self.vector_fields = vector_fields
        self.n_components = n_components

        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

    def fit(self):
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
            self.dataset_id, select_fields=self.vector_fields, include_vector=True
        )

        return self.model.fit_transform_documents(
            vector_field=self.vector_fields[0],
            documents=documents,
            alias=self.alias,
            dims=self.n_components,
        )
