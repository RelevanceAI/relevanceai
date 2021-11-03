# -*- coding: utf-8 -*-
import sys
import time

from loguru import logger

from dataclasses import dataclass

# from ivis import Ivis

from vecdb.base import Base
from api.datasets import Datasets

from typing import Literal, List


@dataclass
class Projection(Base):
    """Projection Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        dataset_id: str,
    ):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        self.dataset_id = dataset_id
        self.documents = self._retrieve_documents()

    def _retrieve_documents(self, number_of_documents=1000):
        dataset = Datasets(self.project, self.api_key, self.base_url)
        page_size = 1000
        resp = dataset.documents.list(
            self.dataset_id, page_size=page_size
        )  # Initial call
        _cursor = resp["cursor"]
        data = []
        while _cursor:
            resp = dataset.documents.list(
                self.dataset_id,
                page_size=page_size,
                cursor=_cursor,
                include_vector=True,
                verbose=True,
            )
            _data = resp["documents"]
            _cursor = resp["cursor"]
            if (_data is []) or (_cursor is []):
                break
            data += _data
            if number_of_documents and (len(data) >= int(number_of_documents)):
                break
        return data

    # def dr(
    #     self,
    #     vector_field: str,
    #     point_label: List[str],
    #     hover_label: List[str],
    #     colour_label: str,
    #     dr_method: Literal['ivis'] = 'ivis',
    #     sample: List = None,
    # ):
    #     """
    #     Dim reduction method
    #     """
    #     pass
