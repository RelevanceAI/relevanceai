"""
Pandas like dataset API
"""
from base64 import encode
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

    def vectorize(
        self,
        image_fields: List[str] = [],
        text_fields: List[str] = [],
        image_encoder=None,
        text_encoder=None,
    ) -> dict:
        if image_fields and image_encoder is None:
            try:
                from vectorhub.bi_encoders.text_image.torch import Clip2Vec

                image_encoder = Clip2Vec()
                image_encoder.encode = image_encoder.encode_image
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default image encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[clip]` to install Clip2Vec."
                )

        if text_fields and text_encoder is None:
            try:
                from vectorhub.encoders.text.tfhub import USE2Vec

                text_encoder = USE2Vec()
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default text encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[encoders-text-tfhub]` to install USE2Vec."
                )

        def create_encoder_function(ftype: str, fields: List[str], encoder):
            if not all(map(lambda field: field in self.schema, fields)):
                raise ValueError(f"Invalid {ftype} field detected")

            if hasattr(encoder, "encode_documents"):

                def encode_documents(documents):
                    return encoder.encode_documents(fields, documents)

            else:

                def encode_documents(documents):
                    return encoder(documents)

            return encode_documents

        if image_fields:
            image_results = self.pull_update_push(
                self.dataset_id,
                create_encoder_function("image", image_fields, image_encoder),
                select_fields=image_fields,
                filters=[
                    {
                        "field": image_field,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": " ",
                        "strict": "must_or",
                    }
                    for image_field in image_fields
                ],
            )
        else:
            image_results = {}

        if text_fields:
            text_results = self.pull_update_push(
                self.dataset_id,
                create_encoder_function("text", text_fields, text_encoder),
                select_fields=text_fields,
                filters=[
                    {
                        "field": text_field,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": " ",
                        "strict": "must_or",
                    }
                    for text_field in text_fields
                ],
            )
        else:
            text_results = {}

        return {"image": image_results, "text": text_results}


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

        super().__init__(project=project, api_key=api_key, firebase_uid=firebase_uid)
