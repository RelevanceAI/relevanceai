"""
Pandas like dataset API
"""
import pandas as pd

from typing import Callable, List, Union

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

        return self

    @property
    def shape(self):
        return self.get_number_of_documents(dataset_id=self.dataset_id)

    def __getitem__(self, index, number_of_documents=20):
        return self.get_documents(
            self.dataset_id,
            select_fields=[index],
            number_of_documents=number_of_documents,
        )

    def info(self) -> dict:
        health = self.datasets.monitor.health(self.dataset_id)
        schema = self.datasets.schema(self.dataset_id)
        schema = {key: str(value) for key, value in schema.items()}
        info = {
            key: {
                "Non-Null Count": health[key]["missing"],
                "Dtype": schema[key],
            }
            for key in schema.keys()
        }
        dtypes = {
            dtype: list(schema.values()).count(dtype)
            for dtype in set(list(schema.values()))
        }
        info = {"info": info, "dtypes": dtypes}
        return info

    def head(self, n: int = 5, raw_json: bool = False) -> Union[dict, pd.DataFrame]:
        head_documents = self.get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            return pd.DataFrame(head_documents).head(n=n)

    def describe(self) -> dict:
        return self.datasets.facets(self.dataset_id)

    def vectorize(self, field: str, model: Callable) -> None:
        def encode_documents(documents):
            return model.encode_documents([field], documents)

        self.pull_update_push(self.dataset_id, encode_documents)
