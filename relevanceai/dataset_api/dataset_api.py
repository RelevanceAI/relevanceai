"""
Pandas like dataset API
"""
from typing import Union, List, Dict

from relevanceai.analytics_funcs import track
from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_export import Export
from relevanceai.dataset_api.dataset_stats import Stats
from relevanceai.dataset_api.dataset_operations import Operations
from relevanceai.dataset_api.dataset_series import Series
from relevanceai.dataset_api.dataset_search import Search

# from relevanceai.dataset_api.dataset_dr import DR


class Dataset(Export, Stats, Operations):
    """Dataset class"""

    @track
    def __init__(
        self,
        project: str,
        api_key: str,
        dataset_id: str,
        firebase_uid: str,
        fields: list = [],
        image_fields: List[str] = [],
        audio_fields: List[str] = [],
        highlight_fields: Dict[str, List] = {},
        text_fields: List[str] = [],
        **kw,
    ):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid
        self.fields = fields
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.audio_fields = audio_fields
        self.highlight_fields = highlight_fields
        self.text_fields = text_fields

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
        )
        self.search = Search(
            project=project,
            api_key=api_key,
            fields=fields,
            dataset_id=dataset_id,
            firebase_uid=firebase_uid,
        )

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


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)
