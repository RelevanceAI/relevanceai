from typing import List, Dict, Any, Optional, Union

from relevanceai.operations_new.base import OperationBase
from relevanceai.operations_new.dr.models.base import DimReductionModelBase


class DimReductionBase(OperationBase):

    model: DimReductionModelBase
    fields: List[str]
    alias: Union[str, None]

    def __init__(
        self,
        vector_fields: List[str],
        alias: Union[str, None],
        model: Union[str, DimReductionModelBase],
        n_components: int,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.vector_fields = vector_fields
        self.alias = alias
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
        if isinstance(model, str):
            if model == "pca":
                from relevanceai.operations_new.dr.models.pca import PCAModel

                model = PCAModel(
                    n_components=n_components,
                    alias=alias,
                    **kwargs,
                )

            elif model == "ivis":
                from relevanceai.operations_new.dr.models.ivis import IvisModel

                model = IvisModel(
                    n_components=n_components,
                    alias=alias,
                    **kwargs,
                )

            elif model == "umap":
                from relevanceai.operations_new.dr.models.umap import UMAPModel

                model = UMAPModel(
                    n_components=n_components,
                    alias=alias,
                    **kwargs,
                )

            elif model == "tsne":
                from relevanceai.operations_new.dr.models.tsne import TSNEModel

                model = TSNEModel(
                    n_components=n_components,
                    alias=alias,
                    **kwargs,
                )

            else:
                raise ValueError(
                    "relevanceai currently does not support this model as a string. the current supported models are [pca, tsne, umpa, ivis]"
                )

            return model

        elif isinstance(model, DimReductionModelBase):
            return model

        else:
            raise ValueError(
                "dim reduction model provided should be either a string or inherit from DimReductionModelBase"
            )

    def run(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        updated_documents = documents

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
            for document in updated_documents
        ]

        return documents
