from abc import abstractmethod
from typing import Any, Dict, List

from relevanceai.utils import DocUtils
from tqdm.auto import tqdm


class VectorizeModelBase(DocUtils):
    model_name: str
    has_printed_vector_field_name: dict = {}

    def _get_model_name(self, url):
        model_name = "_".join(url.split("/google/")[-1].split("/")[:-1])
        return model_name

    def vector_name(self, field, output_field: str = None):
        if output_field is not None:
            return output_field
        return f"{field}_{self.model_name}_vector_".replace("/", "_")

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def bulk_encode(self, *args, **kwargs):
        pass

    def print_vector_field_name(self, field: str, output_field: str = None):
        # Store a simple dictionary to test vectorizing
        if field not in self.has_printed_vector_field_name:
            tqdm.write(
                f"Vector field is `{self.vector_name(field, output_field=output_field)}`"
            )
            self.has_printed_vector_field_name[field] = True

    def encode_documents(
        self,
        documents: List[Dict[str, Any]],
        fields: List[str],
        output_fields: List = None,
    ):
        for i, field in enumerate(fields):
            if output_fields is not None:
                self.print_vector_field_name(field, output_fields[i])
            else:
                self.print_vector_field_name(field)
            values = self.get_field_across_documents(field=field, docs=documents)
            vectors = self.bulk_encode(values)
            if output_fields is not None:
                self.set_field_across_documents(
                    field=self.vector_name(field, output_fields[i]),
                    values=vectors,
                    docs=documents,
                )
            else:
                self.set_field_across_documents(
                    field=self.vector_name(field),
                    values=vectors,
                    docs=documents,
                )

        return documents
