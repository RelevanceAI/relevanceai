from abc import abstractmethod
from pydoc import doc
from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class VectorizeModelBase(DocUtils):
    model_name: str
    has_printed_vector_field_name: dict = {}

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

    def print_vector_field_name(self, field: str):
        # Store a simple dictionary to test vectorizing
        if field not in self.has_printed_vector_field_name:
            print(f"Vector field is `{self.vector_name(field)}`")
            self.has_printed_vector_field_name[field] = True

    def encode_documents(
        self, documents: List[Dict[str, Any]], fields: List[str], bias_value=0
    ):
        for field in fields:
            self.print_vector_field_name(field)
            values = self.get_field_across_documents(field=field, docs=documents)
            vectors = self.bulk_encode(values)

            self.set_field_across_documents(
                field=self.vector_name(field),
                values=vectors,
                docs=documents,
            )

        return documents
