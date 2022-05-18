from typing import List, Dict, Any

from relevanceai.operations_new.base import OperationBase
from relevanceai.operations_new.dim_reduction.models.base import DimReductionModelBase


class DimReductionBase(OperationBase):

    models: List[DimReductionModelBase]
    fields: List[str]

    def __init__(self, fields: List[str], models: List[Any]):
        self.fields = fields
        self.models = [self._get_model(model) for model in models]

    def _get_model(self, model) -> DimReductionModelBase:
        raise NotImplementedError

    def run(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        updated_documents = documents

        for model in self.models:
            for vector_field in self.fields:
                vectors = self.get_field_across_documents(
                    field=vector_field, docs=documents
                )
                reduced_vectors = model.encode(vectors)
                reduced_vector_name = model.vector_name(vector_field)
                self.set_field_across_documents(
                    field=reduced_vector_name, values=reduced_vectors, docs=documents
                )

        # removes unnecessary info for updated_where
        updated_documents = [
            {key: value for key, value in document.items() if key not in self.fields}
            for document in updated_documents
        ]

        return documents
