"""
Pandas like dataset API
"""
import pandas as pd

from typing import Union
from typing import List

class Dataset():
    def __init__(self, client) -> None:
        self.client = client

    def __call__(self, dataset_id: Union[list, str], image_fields: List = [], text_fields: List = [], audio_fields: List = [], output_format: str = 'pandas') -> None:
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.output_format = output_format

        return self

    def info(self) -> None:
        health = self.client.datasets.monitor.health(self.dataset_id)
        schema = self.client.datasets.schema(self.dataset_id)
        
        info = {key: {'missing': health[key]['missing'], 'dtype': schema[key]} for key in health.keys()}
        for key, value in info.items():
            print(key, value)
    
    def head(self):
        # TODO
        return

    def stats(self):
        # TODO
        return

    def describe(self):
        # TODO
        return