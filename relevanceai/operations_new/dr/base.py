from typing import List, Dict, Any, Optional, Union

from relevanceai.operations_new.run import OperationRun
from relevanceai.operations_new.dr.models.base import DimReductionModelBase


class DimReductionBase(OperationRun):

    model: DimReductionModelBase
    fields: List[str]
    alias: str

    def __init__(
        self,
        vector_fields: List[str],
        model: Union[str, DimReductionModelBase],
        n_components: int,
        alias: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.vector_fields = vector_fields

        # TODO: Add a get ailas method
        self.alias = alias  # type: ignore

        if model_kwargs is None:
            model_kwargs = {}

        self.model = self._get_model(
            model=model,
            n_components=n_components,
            alias=alias,
            **model_kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def name(self):
        return "dimensionality_reduction"

    def _get_model(
        self,
        model: Union[str, DimReductionModelBase],
        n_components: int,
        alias: Union[str, None],
        **kwargs,
    ) -> DimReductionModelBase:
        """It returns a model.

        Parameters
        ----------
        model : Union[str, DimReductionModelBase]
            Union[str, DimReductionModelBase]
        n_components : int
            int
        alias : Union[str, None]
            Union[str, None]

        """
        import inspect

        if isinstance(model, str):
            model = self._get_model_from_string(
                model, n_components=n_components, alias=alias  # type: ignore
            )
            return model
        elif isinstance(model, DimReductionModelBase):
            return model
        else:
            raise ValueError(
                "dim reduction model provided should be either a string or inherit from DimReductionModelBase"
            )

    def _get_model_from_string(
        self, model: str, n_components: int, alias: str, **kwargs
    ) -> DimReductionModelBase:
        if model == "pca":
            from relevanceai.operations_new.dr.models.pca import PCAModel

            mapped_model = PCAModel(
                n_components=n_components,
                alias=alias,
                **kwargs,
            )

        elif model == "ivis":
            from relevanceai.operations_new.dr.models.ivis import IvisModel

            mapped_model = IvisModel(
                n_components=n_components,
                alias=alias,
                **kwargs,
            )

        elif model == "umap":
            from relevanceai.operations_new.dr.models.umap import UMAPModel

            mapped_model = UMAPModel(
                n_components=n_components,
                alias=alias,
                **kwargs,
            )

        elif model == "tsne":
            from relevanceai.operations_new.dr.models.tsne import TSNEModel

            mapped_model = TSNEModel(
                n_components=n_components,
                alias=alias,
                **kwargs,
            )

        else:
            raise ValueError(
                "relevanceai currently does not support this model as a string. the current supported models are [pca, tsne, umap, ivis]"
            )
        return mapped_model

    def transform(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        concat_vectors: List[List[float]] = [[] for _ in range(len(documents))]

        for vector_field in self.vector_fields:
            vectors = self.get_field_across_documents(
                field=vector_field, docs=documents
            )
            for vector, concat_vector in zip(vectors, concat_vectors):
                concat_vector.extend(vector)

        reduced_vectors = self.model.fit_transform(concat_vectors)
        reduced_vector_name = self.model.vector_name(self.vector_fields)
        self.set_field_across_documents(
            field=reduced_vector_name, values=reduced_vectors, docs=documents
        )

        # removes unnecessary info for updated_where
        updated_documents = [
            {
                key: value
                for key, value in document.items()
                if key not in self.vector_fields
            }
            for document in documents
        ]

        return documents
