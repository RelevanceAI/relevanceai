from typing import Optional, Tuple, List

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.utils.decorators.version import added, beta
from relevanceai.utils.logger import FileLogger


class _VectorizeHelper(APIClient):
    def __init__(self, dataset_id: str, **kwargs):
        self.dataset_id = dataset_id
        super().__init__(**kwargs)

    def _check_vector_existence(
        self, schema: dict, fields: List[str], encoder
    ) -> Tuple[list, list, list]:
        """
        This function distinguishes new and existing fields, as well as colect
        metadata.

        Parameters
        ----------
        schema: dict
            The schema of a dataset

        fields: List[str]
            The fields to be distinguished.

        encoder_name:
            The name of the encoder to be applied on the fields.
        """
        new_fields = []
        metadata_additions = []
        existing_vectors = []

        if encoder is not None:
            for field in fields:
                vector_parts = (field, encoder.__name__, "vector_")
                vector = "_".join(vector_parts)
                if vector not in schema:
                    new_fields.append(field)
                    metadata_additions.append(vector_parts)
                else:
                    existing_vectors.append(vector)
                    print(
                        f"Since '{vector}' already exists, its construction "
                        + "will be skipped."
                    )

        return new_fields, metadata_additions, existing_vectors

    def _update_vector_metadata(self, new_vector_metadata: List[tuple]) -> None:
        """
        Given the new vector metdata information, updates the Dataset
        metadata.

        Parameters
        ----------
        new_vector_metadata: List[tuple]
            A list of tuples containing information about the newly created
            vectors.
        """
        updated_metadata = self.datasets.metadata(self.dataset_id)["results"]
        if "_vector_" not in updated_metadata:
            updated_metadata["_vector_"] = {}

        for new_data in new_vector_metadata:
            field, model_name, _ = new_data

            if field not in updated_metadata["_vector_"]:
                updated_metadata["_vector_"][field] = {}

            updated_metadata["_vector_"][field][model_name] = "_".join(new_data)

        self.upsert_metadata(updated_metadata)


class VectorizeOps(_VectorizeHelper):
    def __init__(
        self,
        credentials: Credentials,
        dataset_id: str,
    ):
        self.dataset_id = dataset_id
        super().__init__(dataset_id=dataset_id, credentials=credentials)

    @beta
    @added(version="1.2.0")
    def vectorize(
        self,
        image_fields: Optional[List[str]] = None,
        text_fields: Optional[List[str]] = None,
        image_encoder=None,
        text_encoder=None,
        log_file: str = "vectorize.logs",
    ) -> dict:
        """
        Parameters
        ----------
        image_fields: List[str]
            A list of image fields to vectorize

        text_fields: List[str]
            A list of text fields to vectorize

        image_encoder
            A deep learning image encoder from the vectorhub library. If no
            encoder is specified, a default encoder (Clip2Vec) is loaded.

        text_encoder
            A deep learning text encoder from the vectorhub library. If no
            encoder is specified, a default encoder (USE2Vec) is loaded.

        Returns
        -------
        dict
            If the vectorization process is successful, this dict contains
            the added vector names. Else, the dict is the request result
            containing error information.

        Example
        -------
        .. code-block::

            from relevanceai import Client
            from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec

            text_model = SentenceTransformer2Vec("all-mpnet-base-v2 ")

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            df.vectorize(
                image_fields=["image_field_1", "image_field_2"],
                text_fields=["text_field"],
                text_model=text_model
            )

        """
        schema = self.datasets.schema(self.dataset_id)
        if not image_fields and not text_fields:
            raise ValueError("'image_fields' and 'text_fields' both cannot be empty.")

        image_fields = [] if image_fields is None else image_fields
        text_fields = [] if text_fields is None else text_fields

        fields = image_fields + text_fields

        # we should raise an error here to ensure that if there are any missing fields
        # users know immediately
        foreign_fields = []
        for field in fields:
            if field not in schema:
                foreign_fields.append(field)
        else:
            if foreign_fields:
                raise ValueError(
                    f"The following fields are invalid: {', '.join(foreign_fields)}"
                )

        if image_fields and image_encoder is None:
            try:
                with FileLogger(log_file):
                    from vectorhub.bi_encoders.text_image.torch import Clip2Vec

                    image_encoder = Clip2Vec()
                    image_encoder.encode = image_encoder.encode_image
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default image encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[clip]` to install Clip2Vec."
                )
        if image_fields and not hasattr(image_encoder, "encode_documents"):
            raise AttributeError(
                f"{image_encoder} is missing attribute 'encode_documents'"
            )

        (
            new_image_fields,
            image_metadata,
            existing_image_vectors,
        ) = self._check_vector_existence(schema, image_fields, image_encoder)

        if text_fields and text_encoder is None:
            try:
                with FileLogger(log_file):
                    from vectorhub.encoders.text.tfhub import USE2Vec

                    text_encoder = USE2Vec()
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Default text encoder not found. "
                    "Please install vectorhub with `python -m pip install "
                    "vectorhub[encoders-text-tfhub]` to install USE2Vec."
                )
        if text_fields and not hasattr(text_encoder, "encode_documents"):
            raise AttributeError(
                f"{text_encoder} is missing attribute 'encode_documents'"
            )

        (
            new_text_fields,
            text_metadata,
            existing_text_vectors,
        ) = self._check_vector_existence(schema, text_fields, text_encoder)

        if new_image_fields or new_text_fields:

            def dual_encoder_function(documents):
                updated_documents = []
                if image_encoder is not None:
                    updated_documents.extend(
                        image_encoder.encode_documents(new_image_fields, documents)
                    )
                if text_encoder is not None:
                    updated_documents.extend(
                        text_encoder.encode_documents(new_text_fields, documents)
                    )

                return updated_documents

            old_schema = schema.keys()

            results = self.pull_update_push(
                self.dataset_id,
                dual_encoder_function,
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
            )

            new_schema = schema.keys()

            added_vectors = list(new_schema - old_schema)
        else:
            results = {"failed_documents": []}
            added_vectors = []

        if not results["failed_documents"]:
            if added_vectors:
                print("✅ All documents inserted/edited successfully.")
                self._update_vector_metadata(image_metadata + text_metadata)

            if len(added_vectors) == 1:
                text = "The following vector was added: "
            else:
                text = "The following vectors were added: "
            print(text + ", ".join(added_vectors))

            return {
                "added_vectors": added_vectors,
                "skipped_vectors": existing_image_vectors + existing_text_vectors,
            }
        else:
            print("❗Few errors with vectorizing documents. Please check logs.")
            return results
