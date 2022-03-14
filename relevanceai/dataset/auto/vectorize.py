from typing import Optional, List
from relevanceai.dataset.crud.dataset_write import Write
from relevanceai.package_utils.version_decorators import introduced_in_version, beta
from relevanceai.package_utils.logger import FileLogger


class Vectorize(Write):
    @beta
    @introduced_in_version("1.2.0")
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
        if not image_fields and not text_fields:
            raise ValueError("'image_fields' and 'text_fields' both cannot be empty.")

        image_fields = [] if image_fields is None else image_fields
        text_fields = [] if text_fields is None else text_fields

        fields = image_fields + text_fields

        foreign_fields = []
        for field in fields:
            if field not in self.schema:
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
        else:
            new_image_fields = []
            existing_image_vectors = []
            for image_field in image_fields:
                vector = f"{image_field}_{image_encoder.__name__}_vector_"
                if vector not in self.schema:
                    new_image_fields.append(image_field)
                else:
                    existing_image_vectors.append(vector)
                    print(
                        f"Since '{vector}' already exists, its construction will be skipped."
                    )

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
        else:
            new_text_fields = []
            existing_text_vectors = []
            for text_field in text_fields:
                vector = f"{text_field}_{text_encoder.__name__}_vector_"
                if vector not in self.schema:
                    new_text_fields.append(text_field)
                else:
                    existing_text_vectors.append(vector)
                    print(
                        f"Since '{vector}' already exists, its construction will be skipped."
                    )

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

            old_schema = self.schema.keys()

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

            new_schema = self.schema.keys()

            added_vectors = list(new_schema - old_schema)
        else:
            results = {"failed_documents": []}
            added_vectors = []

        if not results["failed_documents"]:
            if added_vectors:
                print("✅ All documents inserted/edited successfully.")

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
