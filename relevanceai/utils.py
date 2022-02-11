from typing import List, Dict

from doc_utils import DocUtils
from relevanceai.base import _Base
from relevanceai.api.endpoints.client import APIClient


class Utils(APIClient, _Base, DocUtils):
    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)

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

    def _remove_empty_vector_fields(self, documents, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        return [d for d in documents if d.get(vector_field)]

    def _convert_id_to_string(self, documents):
        self.set_field_across_documents(
            "_id", [str(i["_id"]) for i in documents], documents
        )

    def _are_fields_in_schema(self, fields, dataset_id, schema=None):
        """
        Check fields are in schema
        """
        if schema is None:
            schema = self.datasets.schema(dataset_id)
        invalid_fields = []
        for i in fields:
            if i not in schema:
                invalid_fields.append(i)
        if len(invalid_fields) > 0:
            raise ValueError(
                f"{', '.join(invalid_fields)} are invalid fields. They are not in the dataset schema."
            )
        return
