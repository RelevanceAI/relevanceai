from typing import List, Dict, Any, Optional, Union

from relevanceai.operations_new.run import OperationRun
from relevanceai.operations_new.scaling.models.base import ScalerModelBase


class ScalerBase(OperationRun):

    model: ScalerModelBase
    fields: List[str]
    alias: str

    def __init__(
        self,
        vector_fields: List[str],
        model: Union[str, ScalerModelBase],
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
            alias=alias,
            model_kwargs=model_kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def name(self):
        return "scaing"

    def _get_model(
        self,
        model: Union[str, ScalerModelBase],
        alias: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ) -> ScalerModelBase:

        from relevanceai.operations_new.scaling.models.sklearn import SKLearnScaler

        mapped_model = SKLearnScaler(
            model=model,
            alias=alias,
            model_kwargs=model_kwargs,
        )

        return mapped_model

    def transform(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        scaled_vector_names = []

        for vector_field in self.vector_fields:
            vectors = self.get_field_across_documents(
                field=vector_field, docs=documents
            )

            reduced_vectors = self.model.fit_transform(vectors)
            scaled_vector_name = self.model.vector_name(vector_field)

            if scaled_vector_name in self.vector_fields:
                raise ValueError(
                    "Alias is already being used, Please set a different alias"
                )

            scaled_vector_names.append(scaled_vector_name)

            self.set_field_across_documents(
                field=scaled_vector_name,
                values=reduced_vectors,
                docs=documents,
            )

        # removes unnecessary info for updated_where
        updated_documents = [
            {
                key: value
                for key, value in document.items()
                if key in ["_id"] + scaled_vector_names
            }
            for document in documents
        ]

        return updated_documents
