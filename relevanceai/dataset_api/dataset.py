"""
Pandas like dataset API
"""
import pandas as pd

from typing import Union
from typing import List

from tabulate import tabulate


class Dataset:
    def __init__(self, client) -> None:
        self.client = client

    def __call__(
        self,
        dataset_id: Union[list, str],
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

    def info(self) -> None:
        health = self.client.datasets.monitor.health(self.dataset_id)
        schema = self.client.datasets.schema(self.dataset_id)

        info = [
            {
                "#": index,
                "key": key,
                "missing": health[key]["missing"],
                "dtype": schema[key],
            }
            for index, key in enumerate(health.keys())
        ]
        headers = list(info[0].keys())
        table_data = [list(schema_info.values()) for schema_info in info]
        print(tabulate(table_data, headers=headers))

    def head(self) -> None:
        head_docs = self.client.get_documents(self.dataset_id, number_of_documents=5)
        head_df = pd.DataFrame(head_docs).head()
        head_df = head_df.drop(["_id", "insert_date_"], axis=1)
        return head_df

    def stats(self):
        # TODO
        raise NotImplementedError

    def describe(self):
        # TODO
        raise NotImplementedError
