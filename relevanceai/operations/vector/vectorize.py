from typing import Union, Optional, List, Any

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.constants import IMG_EXTS
from relevanceai.constants import Messages

from relevanceai.utils.decorators import silence
from relevanceai.utils.logger import FileLogger


class VectorizeOps(APIClient):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
        image_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
        image_encoder: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
        log_file: str = "vectorize.logs",
    ):
        super().__init__(dataset_id=dataset_id, credentials=credentials)

        self.dataset_id = dataset_id

        self.schema = self._get_schema()
        self.detailed_schema = self._get_detailed_schema()

        image_fields = [] if image_fields is None else image_fields
        if not image_fields:
            self.image_fields = [
                field
                for field, dtype in self.detailed_schema.items()
                if dtype == "_image_"
            ]
        else:
            self.image_fields = image_fields

        text_fields = [] if text_fields is None else text_fields
        if not text_fields:
            self.text_fields = [
                field
                for field, dtype in self.detailed_schema.items()
                if dtype == "_text_"
            ]
        else:
            self.text_fields = text_fields

        self.log_file = log_file

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        print("Initialising default encoders...")
        self._init_encoders()
        print("Done")

    def __call__(self, *args, **kwargs):
        return self.operate(*args, **kwargs)

    @silence
    def _init_encoders(self):
        self._init_image_encoder()
        self._init_text_encoder()

    def _init_image_encoder(self) -> None:
        if self.image_fields and self.image_encoder is None:
            try:
                with FileLogger(self.log_file):
                    from vectorhub.bi_encoders.text_image.torch import Clip2Vec

                    self.image_encoder = Clip2Vec(jit=False)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default image encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[clip]` to install Clip2Vec."
                )
        if self.image_fields and not hasattr(self.image_encoder, "encode_documents"):
            raise AttributeError(
                f"{self.image_encoder} is missing attribute 'encode_documents'"
            )

    def _init_text_encoder(self) -> None:
        if self.text_fields and self.text_encoder is None:
            try:
                with FileLogger(self.log_file):
                    from vectorhub.encoders.text.tfhub import USE2Vec

                    self.text_encoder = USE2Vec()
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default text encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[encoders-text-tfhub]` to install USE2Vec."
                )
        if self.text_fields and not hasattr(self.text_encoder, "encode_documents"):
            raise AttributeError(
                f"{self.text_encoder} is missing attribute 'encode_documents'"
            )

    def _get_dtype(self, value: Any) -> Union[str, None]:
        if isinstance(value, str):
            if "https://" in value:
                if any(img_ext in value for img_ext in IMG_EXTS):
                    return "_image_"

            elif "$" in value:
                try:
                    float(value.replace("$", ""))
                    return "numeric"

                except:
                    return "_text_"

            elif len(value.strip().split(" ")) > 1:
                return "_text_"

        return None

    def _get_schema(self):
        return self.datasets.schema(self.dataset_id)

    def _get_detailed_schema(self):
        schema = self.datasets.schema(dataset_id=self.dataset_id)

        text_fields = [column for column, dtype in schema.items() if dtype == "text"]

        for column in text_fields:
            value = self._get_documents(
                dataset_id=self.dataset_id,
                select_fields=[column],
                filters=[
                    {
                        "field": column,
                        "filter_type": "exists",
                        "condition": "==",
                        "condition_value": "",
                    }
                ],
                number_of_documents=1,
            )[0][column]
            dtype = self._get_dtype(value)
            if dtype is not None:
                schema[column] = dtype

        return schema

    def _validate_fields(self, fields):
        foreign_fields = []
        for field in fields:
            if field not in self.schema:
                foreign_fields.append(field)
        else:
            if foreign_fields:
                raise ValueError(
                    f"The following fields are invalid: {', '.join(foreign_fields)}"
                )

    @staticmethod
    def _encode_documents(
        documents,
        image_encoder,
        text_encoder,
        image_fields,
        text_fields,
    ):
        if image_encoder is not None:
            updated_documents = image_encoder.encode_documents(
                image_fields,
                documents,
            )

        if text_encoder is not None:
            updated_documents = text_encoder.encode_documents(
                text_fields,
                documents,
            )

        return updated_documents

    def operate(
        self,
        image_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
    ) -> dict:

        image_fields = self.image_fields if image_fields is None else image_fields
        text_fields = self.text_fields if text_fields is None else text_fields

        fields = image_fields + text_fields
        self._validate_fields(fields)

        old_schema = self.schema.keys()

        results = self.pull_update_push(
            dataset_id=self.dataset_id,
            update_function=self._encode_documents,
            retrieve_chunk_size=20,
            select_fields=fields,
            filters=[
                {
                    "field": field,
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": " ",
                    "strict": "must_or",
                }
                for field in fields
            ],
            show_progress_bar=True,
            updating_args=dict(
                image_encoder=self.image_encoder,
                text_encoder=self.text_encoder,
                image_fields=self.image_fields,
                text_fields=self.text_fields,
            ),
        )

        new_schema = self._get_schema().keys()

        added_vectors = list(new_schema - old_schema)

        if not results["failed_documents"]:
            if added_vectors:
                print(Messages.INSERT_GOOD)

            text = "The following vector fields were added: "
            print(text + ", ".join(added_vectors))

            return {
                "added_vectors": added_vectors,
            }
        else:
            print(Messages.INSERT_BAD)
            return results
