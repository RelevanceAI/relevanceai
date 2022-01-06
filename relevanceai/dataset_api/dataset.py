"""
Pandas like dataset API
"""
import pandas as pd

from relevanceai.http_client import Client

from typing import Union
from typing import List

class Dataset():
    def __init__(self, project: str, api_key: str) -> None:
        self.project = project
        self.api_key = api_key

    def __call__(self, dataset_id: Union[list, str], image_fields: List = [], text_fields: List = [], audio_fields: List = [], output_format: str = 'pandas') -> None:
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.output_format = output_format

    # @property ????
    def info(self):
        health = Client().datasets.monitor.health(self.dataset_id)
        schema = Client().datasets.schema(self.dataset_id)
        return health
    
    def head(self):
        # TODO
        return

    def stats(self):
        # TODO
        return

    def describe(self):
        # TODO
        return