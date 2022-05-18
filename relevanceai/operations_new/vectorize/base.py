from typing import List, Dict, Any

from abc import abstractmethod

from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.operations_new.base import OperationBase


class VectorizeOperationBase(OperationBase):

    models: List[VectorizeModelBase]
    fields: List[str]

    def __init__(self, fields: List[str], models: List[Any]):
        self.fields = fields
        self.models = [self._get_model(model) for model in models]

    @abstractmethod
    def _get_model(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """It takes a list of documents, and for each document, it runs the document through each of the
        models in the pipeline, and returns the updated documents.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List[Dict[str, Any]]

        Returns
        -------
            A list of dictionaries.

        """

        updated_documents = documents

        for model in self.models:
            updated_documents = model.encode_documents(
                documents=updated_documents,
                fields=self.fields,
            )

        # removes unnecessary info for updated_where
        updated_documents = [
            {key: value for key, value in document.items() if key not in self.fields}
            for document in updated_documents
        ]

        return updated_documents
