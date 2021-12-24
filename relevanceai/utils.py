from doc_utils import DocUtils
from relevanceai.base import _Base
from relevanceai.api.client import BatchAPIClient
from typing import List, Dict


class Utils(BatchAPIClient, _Base, DocUtils):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def _is_valid_vector_name(self, dataset_id, vector_name: str) -> bool:
        """
        Check vector field name is valid
        """
        vector_fields = self.get_vector_fields(dataset_id)
        schema = self.datasets.schema(dataset_id)
        if vector_name in schema.keys():
            if vector_name in vector_fields:
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(f"{vector_name} is not in the {dataset_id} schema")

    def _is_valid_label_name(self, dataset_id, label_name: str) -> bool:
        """
        Check vector label name is valid. Checks that it is either numeric or text
        """
        schema = self.datasets.schema(dataset_id)
        if label_name == "_id":
            return True
        if label_name in list(schema.keys()):
            if schema[label_name] in ["numeric", "text"]:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(f"{label_name} is not in the {dataset_id} schema")

    def _remove_empty_vector_fields(self, docs, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        return [d for d in docs if d.get(vector_field)]
