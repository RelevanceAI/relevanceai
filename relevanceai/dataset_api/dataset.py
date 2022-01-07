"""
Pandas like dataset API
"""
import pandas as pd

from typing import Union
from typing import List

from relevanceai.api.client import BatchAPIClient


class Dataset(BatchAPIClient):
    def __init__(self, project, api_key) -> None:
        super().__init__(project, api_key)
        self.project = project
        self.api_key = api_key

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
        output_format: str = "pandas",
    ):
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.output_format = output_format

        self.health = self.datasets.monitor.health(self.dataset_id)
        self.schema = self.datasets.schema(self.dataset_id)

        self.docs = self.get_all_documents(self.dataset_id)
        self.df = pd.DataFrame(self.docs)

        return self

    @property
    def shape(self):
        return self.df.shape

    def info(self) -> None:
        return self.df.info()

    def head(self, raw_json=False) -> None:
        return self.df.head()

    def describe(self):
        return self.df.describe()
