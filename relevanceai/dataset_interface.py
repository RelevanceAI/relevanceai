import pandas as pd
from typing import Dict, List, Optional, Union

from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.export.interface import Export
from relevanceai.dataset.auto.dataset_operations import Operations
from relevanceai.dataset.crud.dataset_series import Series
from relevanceai.dataset.search.search import Search
from relevanceai.dataset.vis.plot import Plot

# TODO: Add game dataset
_GLOBAL_DATASETS = ["_mock_dataset_"]


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


class Dataset(Export, Plot, Operations, Search):
    @track
    def __init__(
        self,
        project: str,
        api_key: str,
        dataset_id: str,
        firebase_uid: str,
        fields: Optional[list] = None,
        image_fields: Optional[List[str]] = None,
        audio_fields: Optional[List[str]] = None,
        highlight_fields: Optional[Dict[str, List]] = None,
        text_fields: Optional[List[str]] = None,
        **kw,
    ):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid
        self.fields = [] if fields is None else fields
        self.dataset_id = dataset_id
        self.image_fields = [] if image_fields is None else image_fields
        self.audio_fields = [] if audio_fields is None else audio_fields
        self.highlight_fields = {} if highlight_fields is None else highlight_fields
        self.text_fields = [] if text_fields is None else text_fields
        self.firebase_uid = firebase_uid

        super().__init__(
            project=project,
            api_key=api_key,
            firebase_uid=firebase_uid,
            fields=fields,
            dataset_id=dataset_id,
            image_fields=image_fields,
            audio_fields=audio_fields,
            highlight_fields=highlight_fields,
            text_fields=text_fields,
            **kw,
        )
        self.search = Search(
            project=project,
            api_key=api_key,
            firebase_uid=firebase_uid,
        )
        # add global datasets
        if self.dataset_id in _GLOBAL_DATASETS:
            # avoid re-inserting if it already exists
            if self.dataset_id not in self.datasets.list()["datasets"]:
                from relevanceai.package_utils.datasets import mock_documents
                from relevanceai.package_utils.analytics_funcs import fire_and_forget

                @fire_and_forget
                def add_mock_dataset():
                    self.upsert_documents(mock_documents(100))

                add_mock_dataset()

    # def __getattr__(self, attr):
    #     if hasattr(pd.DataFrame, attr):
    #         df = self.to_pandas_dataframe(show_progress_bar=True)
    #         try:
    #             return getattr(df, attr)
    #         except SyntaxError:
    #             raise AttributeError(f"'{attr}' is an invalid attribute")
    #     raise AttributeError(f"'{attr}' is an invalid attribute")

    @track
    def __getitem__(self, field: Union[List[str], str]):
        """
        Returns a Series Object that selects a particular field within a dataset
        Parameters
        ----------
        field: Union[List, str]
            The particular field within the dataset
        Returns
        -------
        Tuple
            (N, C)
        Example
        ---------------
        .. code-block::
            from relevanceai import Client
            client = Client()
            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            field = "sample_field"
            series = df[field]
        """
        if isinstance(field, str):
            return Series(
                project=self.project,
                api_key=self.api_key,
                dataset_id=self.dataset_id,
                firebase_uid=self.firebase_uid,
                field=field,
                image_fields=self.image_fields,
                audio_fields=self.audio_fields,
                highlight_fields=self.highlight_fields,
                text_fields=self.text_fields,
            )
        elif isinstance(field, list):
            return Dataset(
                project=self.project,
                api_key=self.api_key,
                dataset_id=self.dataset_id,
                firebase_uid=self.firebase_uid,
                fields=field,
                image_fields=self.image_fields,
                audio_fields=self.audio_fields,
                highlight_fields=self.highlight_fields,
                text_fields=self.text_fields,
            )
        else:
            raise TypeError("Field needs to be a list or a string.")

    @track
    def launch_search_app(self):
        """
        Launches the link to the search application to start building
        """
        return (
            f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/search"
        )

    @track
    def launch_projector_app(self):
        """
        Launches the link to the projector application to start building
        """
        return f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/projector"
