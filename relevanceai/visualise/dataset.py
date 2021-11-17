# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from dataclasses import dataclass

from relevanceai.base import Base
from relevanceai.api.datasets import Datasets

from typing import List, Union, Dict, Any, Optional
from typing_extensions import Literal

from doc_utils import DocUtils

JSONDict = Dict[str, Any]

@dataclass
class Dataset(Base, DocUtils):
    """Dataset Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        dataset_id: str,
        vector_field: str,
        vector_label: Union[None, str],
        colour_label: Union[None, str],
        hover_label: Union[None, List[str]],
        number_of_documents: Union[None, int] = 1000,
        page_size: int = 1000,
        random_state: int = 0,
    ):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

        self.dataset_id = dataset_id
        self.vector_field = vector_field
        self.logger.info(f'Retrieving {number_of_documents} documents from {dataset_id} ...')

        self.dataset = Datasets(self.project, self.api_key, self.base_url)
        self.random_state = random_state
        
        if hover_label is None: hover_label=[]
        fields = [label for label in ['_id', vector_field, vector_label, colour_label]+hover_label if label] # type: ignore
        self.docs = self._retrieve_documents(dataset_id, fields, number_of_documents, page_size)
        self.vector_fields = self._vector_fields()
        self.docs = self._remove_empty_vector_fields(vector_field)
        self.vector_dim = self.schema[vector_field]['vector']

        
    
    def _retrieve_documents(
        self, 
        dataset_id: str, 
        fields: List[str],
        number_of_documents: Optional[int] = 1000,
        page_size: int = 1000,
    ) -> List[JSONDict]:
        """
        Retrieve all documents from dataset
        """

        if number_of_documents:
            if page_size > number_of_documents or self.random_state != 0:
                page_size = number_of_documents # type: ignore
        else:
            number_of_documents = 999999999999999

        is_random = True if self.random_state != 0 else False
        resp = self.dataset.documents.get_where(
            dataset_id=dataset_id, select_fields=fields, 
            include_vector=True,
            page_size=page_size, is_random=is_random, 
            random_state=self.random_state
            )

        data = resp["documents"]
        
        if (number_of_documents > page_size) and (is_random==False) and (self.random_state==0):
            _cursor = resp["cursor"]
            _page = 0
            while resp:
                self.logger.debug(f'Paginating {_page} page size {page_size} ...')
                resp = self.dataset.documents.get_where(
                    dataset_id=dataset_id,
                    select_fields=fields,
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
            data = data[:number_of_documents]

        self.df, self.detail = self._build_df(data)
        self.docs = data
        return self.docs

    @staticmethod
    def _build_df(data: List[JSONDict]):
        df = pd.DataFrame(data)
        # detail_cols = self.get_fields_across_documents_except(fields=['_vector_', '_id', 'insert_date_'], data=data)
        detail_cols = [ c for c in df.columns if not any(f in c for f in ['_vector_', '_id', 'insert_date_']) ]
        detail = df[detail_cols]
        return df, detail


    def _vector_fields(
        self
    ) -> List[str]:
        """
        Returns list of valid vector fields from dataset schema
        """
        self.schema = self.dataset.schema(dataset_id=self.dataset_id)

        return [k for k, v in self.schema.items()
                if isinstance(v, dict) 
                if 'vector' in v.keys()]
    
    def valid_vector_name(self, vector_name: str) -> bool:
        """
        Check vector field name is valid
        """
        if vector_name in self.schema.keys():
            if vector_name in self.vector_fields:
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(f"{vector_name} is not in the {self.dataset_id} schema")

    def valid_label_name(self, label_name: str) -> bool:
        """
        Check vector label name is valid
        """
        if label_name == '_id': return True
        if label_name in list(self.schema.keys()):
            if  (self.schema[label_name] in ['numeric', 'text']):
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(f"{label_name} is not in the {self.dataset_id} schema")

    def _remove_empty_vector_fields(self, vector_field: str) -> List[JSONDict]:
        """
        Remove documents with empty vector fields
        """
        self.docs = [ d for d in self.docs if d.get(vector_field) ]
        self.df, self.detail = self._build_df(self.docs)
        return self.docs