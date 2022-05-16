from abc import abstractmethod
from pydoc import doc
from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class ModelBase(DocUtils):
    @staticmethod
    def vector_name(field):
        return field + "_vector_"

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def bulk_encode(self, *args, **kwargs):
        pass

    def encode_documents(
        self, documents: List[Dict[str, Any]], fields: List[str], bias_value=0
    ):
        for field in fields:
            values = self.get_field_across_documents(field=field, docs=documents)
            vectors = self.bulk_encode(values)

            self.set_field_across_documents(
                field=self.vector_name(field),
                values=vectors,
                docs=documents,
            )

        return documents
