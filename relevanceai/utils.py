import dataclasses
from enum import Enum
from pathlib import PurePath
from types import GeneratorType
import numpy as np
import collections
from typing import List, Dict
from pydantic.json import ENCODERS_BY_TYPE
import pandas as pd
import math

from doc_utils import DocUtils
from relevanceai.base import _Base
from relevanceai.api.endpoints.client import APIClient


class Utils(APIClient, _Base, DocUtils):
    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def json_encoder(self, obj):
        """
        Converts object so it is json serializable
        """
        # Loop through iterators and convert
        if isinstance(
            obj, (list, set, frozenset, GeneratorType, tuple, collections.deque)
        ):
            encoded_list = []
            for item in obj:
                encoded_list.append(self.json_encoder(item))
            return encoded_list

        # Loop through dictionaries and convert
        if isinstance(obj, dict):
            encoded_dict = {}
            for key, value in obj.items():
                encoded_key = self.json_encoder(key)
                encoded_value = self.json_encoder(value)
                encoded_dict[encoded_key] = encoded_value
            return encoded_dict

        # Custom conversions
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, PurePath):
            return str(obj)
        if isinstance(obj, (str, int, type(None))):
            return obj
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            else:
                return obj
        if type(obj) in ENCODERS_BY_TYPE:
            return ENCODERS_BY_TYPE[type(obj)](obj)

        raise ValueError(f"{obj} ({type(obj)}) cannot be converted to JSON format")

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
