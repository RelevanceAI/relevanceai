from typing import Any, List, Union

from relevanceai.operations_new.base import OperationsBase
from relevanceai.operations_new.vectorize.models.base import ModelBase


class VectorizeOps(OperationsBase):
    tfhub_models: List[str] = []
    sentencetransformer_models: List[str] = []

    def __init__(self, fields: List[str], models: List[Any]):

        self.fields = fields
        self.models = [self._get_model(model) for model in models]

    def _get_model(self, model: Any) -> Union[None, ModelBase]:
        if isinstance(model, str):
            if model in self.tfhub_models:
                return None

            elif model in self.sentencetransformer_models:
                return None

            else:
                raise ValueError("Model not a valid model string")

        elif isinstance(model, ModelBase):
            return model

        else:
            raise ValueError(
                "Model should either be a supported model string or inherit from ModelBase"
            )

    def run(self, documents):

        updated_documents = documents

        for model in self.models:
            updated_documents = model.encode_documents(
                documents=updated_documents,
                fields=self.fields,
            )

        return updated_documents
