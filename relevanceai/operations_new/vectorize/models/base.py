from abc import abstractmethod
from pydoc import doc
from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class VectorizeModelBase(DocUtils):
    model_name: str

    def _get_model_name(self, url):
        model_name = "_".join(url.split("/google/")[-1].split("/")[:-1])
        return model_name

    def vector_name(self, field):
        return f"{field}_{self.model_name}_vector_"

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
