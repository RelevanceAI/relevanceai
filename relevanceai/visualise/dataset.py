# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from dataclasses import dataclass

from relevanceai.base import Base
from api.datasets import Datasets

from typing import List, Union, Dict, Any, Tuple
from typing_extensions import Literal

JSONDict = Dict[str, Any]


@dataclass
class Dataset(Base):
    """Dataset Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        dataset_id: str,
        number_of_documents: Union[None, int] = 1000,
        page_size: int = 1000,
    ):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

        self.dataset_id = dataset_id
        self.logger.info(f'Retrieving {number_of_documents} documents from {dataset_id} ...')

        self.dataset = Datasets(self.project, self.api_key, self.base_url)

        self.data = self._retrieve_documents(dataset_id, number_of_documents, page_size)
        self.vector_fields = self._vector_fields()

    
    def _retrieve_documents(
        self, 
        dataset_id: str, 
        number_of_documents: Union[None, int] = 1000, 
        page_size: int = 1000
    ) -> List[JSONDict]:
        """
        Retrieve all documents from dataset
        """
        
        if (number_of_documents and page_size > number_of_documents): page_size=number_of_documents
        resp = self.dataset.documents.list(
            dataset_id=dataset_id, page_size=page_size
        )  # Initial call
        _cursor = resp["cursor"]
        _page = 0
        data = []
        while resp:
            self.logger.debug(f'Paginating {_page} page size {page_size} ...')
            resp = self.dataset.documents.list(
                dataset_id=dataset_id,
                page_size=page_size,
                cursor=_cursor,
                include_vector=True,
                verbose=True,
            )
            _data = resp["documents"]
            _cursor = resp["cursor"]
            if (_data == []) or (_cursor == []): break
            data += _data 
            if number_of_documents and (len(data) >= int(number_of_documents)): break
            _page += 1
        
        self.df = pd.DataFrame(data)
        metadata_cols = [c for c in self.df.columns 
                            if not any(s in c for s in ['_vector_', '_id', 'insert_date_'])]
        self.metadata = self.df[metadata_cols]

        self.data = data
        return self.data


    def _vector_fields(
        self
    ) -> List[str]:
        """
        Returns list of valid vector fields from datset schema
        """
        self.schema = self.dataset.schema(dataset_id=self.dataset_id)
        return [k for k, v in self.schema.items()
                if isinstance(v, dict) 
                if 'vector' in v.keys()]
    

    def valid_vector_name(self, vector_name: str) -> bool:
        if vector_name in self.schema.keys():
            if (type(self.schema[vector_name]) == dict) and self.schema[vector_name].get('vector'):
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(f"{vector_name} is not in the {self.dataset_name} schema")

    def valid_label_name(self, label_name: str) -> bool:
        if label_name in self.schema.keys():
            if self.schema[label_name] in ['numeric', 'text']:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(f"{label_name} is not in the {self.dataset_name} schema")
