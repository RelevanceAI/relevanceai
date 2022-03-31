from typing import Dict, Union, Optional, List, Any

import numpy as np

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.constants import IMG_EXTS
from relevanceai.constants import Messages

from relevanceai.utils.decorators import log


class VectorizeOps(APIClient):
    log_file: str = "vectorize.logs"

    def __init__(
        self,
        credentials: Credentials,
        encoders: Optional[Dict[str, Any]] = None,
        log_file: str = "vectorize.logs",
    ):
        super().__init__(credentials=credentials)

        self.log_file = log_file
        self.encoders = encoders if encoders is not None else {}

    def __call__(self, *args, **kwargs):
        return self.operate(*args, **kwargs)

    @log(fn=log_file)
    def _get_encoder(self, model: Any) -> Any:
        if isinstance(model, str):
            model = model.lower().replace(" ", "").replace("_", "")
            self.model_name = model
        else:
            self.model_name = str(model.__class__).split(".")[-1].split("'>")[0]
            return model

        if isinstance(model, str):
            if model == "use":
                from vectorhub.encoders.text.tfhub import USE2Vec

                model = USE2Vec()

            elif model == "bert":
                from vectorhub.encoders.text.tfhub import Bert2Vec

                model = Bert2Vec()

            elif model == "labse":
                from vectorhub.encoders.text.tfhub import LaBSE2Vec

                model = LaBSE2Vec()

            elif model == "elmo":
                from vectorhub.encoders.text.tfhub import Elmo2Vec

                model = Elmo2Vec()

            elif model == "clip":
                from vectorhub.bi_encoders.text_image.torch.clip import Clip2Vec

                model = Clip2Vec()
                model.encode = model.encode_image

            elif model == "resnet":
                from vectorhub.encoders.image.fastai import FastAIResnet2Vec

                model = FastAIResnet2Vec()

            elif model == "mobilenet":
                from vectorhub.encoders.image.tfhub import MobileNetV12Vec

                model = MobileNetV12Vec()

        else:
            # TODO: this needs to be referenced from relevance.constants.errors
            raise ValueError("ModelNotSupported")

        return model

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
            values = self._get_documents(
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
            )
            if values:
                value = values[0][column]
            else:
                continue

            dtype = self._get_dtype(value)
            if dtype is not None:
                schema[column] = dtype

        return schema

    def _validate_fields(self, fields: List[str]):
        foreign_fields = []
        for field in fields:
            if field not in self.schema:
                foreign_fields.append(field)
        else:
            if foreign_fields:
                raise ValueError(
                    f"The following fields are invalid: {', '.join(foreign_fields)}"
                )

    def _get_vector_fields(self, fields: List[str]) -> List[str]:
        vector_fields = []

        for field in fields:
            dtype = self.detailed_schema[field].replace("_", "")
            encoder = self.encoders[dtype]
            vector_field = encoder.get_default_vector_field_name(field)
            vector_fields.append(vector_field)

        return vector_fields

    def _get_filters(
        self, fields: List[str], vector_fields: List[str]
    ) -> List[Dict[str, Any]]:

        filters = []

        for i in range(len(fields) ** 2):
            binary_array = [character for character in str(bin(i))][2:]
            mixed_mask = ["0"] * (len(fields) - len(binary_array)) + binary_array
            mask = [int(value) for value in mixed_mask]

            condition_value = [
                {
                    "field": field if mask[index] else vector_field,
                    "filter_type": "exists",
                    "condition": "==" if mask[index] else "!=",
                    "condition_value": "",
                }
                for index, (field, vector_field) in enumerate(
                    zip(fields, vector_fields)
                )
            ]
            filters.append(
                {
                    "filter_type": "or",
                    "condition_value": condition_value,
                }
            )

        return filters

    @staticmethod
    def _encode_documents(
        documents,
        encoders,
        fields,
    ):
        for dtype, encoder in encoders.items():
            dtype_fields = [
                field
                for field, field_type in fields.items()
                if f"_{dtype}_" == field_type
            ]
            updated_documents = encoder.encode_documents(
                dtype_fields,
                documents,
            )

        return updated_documents

    def _reduce(self, vectors: np.ndarray, n_components: int = 512) -> np.ndarray:
        from sklearn.decomposition import PCA

        reducer = PCA(
            n_components=min(vectors.shape[0], vectors.shape[1], n_components)
        )
        reduced = reducer.fit_transform(vectors)
        return reduced

    def _get_fields(self, fields: List[str]) -> Dict[str, str]:
        return {
            field: self.detailed_schema[field]
            for field in fields
            if any(
                dtype in self.detailed_schema[field] for dtype in ["_image_", "_text_"]
            )
        }

    def _get_model_name(self, dtype: str) -> str:
        if dtype == "_text_":
            if "text" in self.encoders:
                return self.encoders["text"]

            else:
                return "use"

        elif dtype == "_image_":
            if "image" in self.encoders:
                return self.encoders["image"]

            else:
                return "clip"

        else:
            raise ValueError

    def _init_encoders(self, fields: List[str]):
        dtypes = [self.detailed_schema[field] for field in fields]

        for dtype in ["image", "text"]:
            if f"_{dtype}_" in dtypes:
                model = self._get_model_name(dtype=f"_{dtype}_")
                self.encoders[dtype] = self._get_encoder(model=model)

    def operate(
        self,
        dataset_id: str,
        fields: List[str],
        show_progress_bar: bool = True,
    ) -> dict:

        self.dataset_id = dataset_id
        self.schema = self._get_schema()
        self.detailed_schema = self._get_detailed_schema()

        if fields is None:
            fields = self.detailed_schema

        field_types = self._get_fields(fields)
        self._validate_fields(list(field_types))
        self._init_encoders(list(field_types))

        vector_fields = self._get_vector_fields(list(field_types))

        old_schema = self.schema.keys()

        filters = self._get_filters(
            fields=list(field_types),
            vector_fields=vector_fields,
        )

        updating_args = dict(
            encoders=self.encoders,
            fields=fields,
        )

        results = self.pull_update_push(
            dataset_id=self.dataset_id,
            retrieve_chunk_size=4,
            update_function=self._encode_documents,
            select_fields=list(fields),
            filters=filters,
            show_progress_bar=show_progress_bar,
            updating_args=updating_args,
            log_to_file=False,
        )

        schema = self._get_schema()

        if "unstructured_document_vector_" not in schema:
            documents = self._get_all_documents(
                dataset_id=self.dataset_id,
                select_fields=vector_fields,
                show_progress_bar=True,
            )
            vectors = np.hstack(
                [
                    np.array(
                        [
                            self.get_field(vector_field, document)
                            if self.is_field(vector_field, document)
                            else [1e-7 for _ in range(schema[vector_field]["vector"])]
                            for document in documents
                        ]
                    )
                    for vector_field in vector_fields
                ]
            )
            if vectors.shape[1] > 512:
                vectors = self._reduce(vectors)

            for index, (document, vector) in enumerate(
                zip(documents, vectors.tolist())
            ):
                documents[index] = {
                    "_id": document["_id"],
                    "unstructured_document_vector_": vector,
                }

            self._update_documents(
                dataset_id=self.dataset_id,
                documents=documents,
                show_progress_bar=True,
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
