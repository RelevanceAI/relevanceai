from typing import Dict, Union, Optional, List, Any

import numpy as np

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.constants import Messages
from relevanceai.constants import IMG_EXTS

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
            try:
                value = values[0][column]
            except:
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

    def _get_numeric_fields(self) -> List[str]:
        numeric_fields = [
            field for field, dtype in self.schema.items() if dtype == "numeric"
        ]
        return numeric_fields

    def _get_filters(
        self, fields: List[str], vector_fields: List[str]
    ) -> List[Dict[str, Any]]:

        """
        Creates the filters necessary to search all documents
        within a dataset that contain fields specified in "fields"
        but do not contain their resepctive vector_fields defined in "vector_fields"

        e.g.
        fields = ["text", "title"]
        vector_fields = ["text_use_vector_", "title_use_vector_"]

        we want to search the dataset where:
        ("text" * ! "text_use_vector_") + ("title" * ! "title_use_vector_")

        Since the current implementation of filtering only accounts for CNF and not DNF boolean logic,
        We must use boolean algebra here to obtain the CNF from a DNF expression.

        CNF = Conjunctive Normal Form (Sum of Products)
        DNF = Disjunctive Normal Form (Product of Sums)

        This means converting the above to:
        ("text" + "title") * ("text" + ! "title_use_vector_") *
        (! "text_use_vector_" + "title") * (! "text_use_vector_" + ! "title_use_vector_")

        Arguments:
            fields: List[str]
                A list of fields within the dataset

            vector_fields: List[str]
                A list of vector_fields, created from the fields given the current encoders.
                These would be present if the fields in "fields" were vectorized

        Returns:
            filters: List[Dict[str, Any]]
                A list of filters.
        """

        filters = []
        if len(fields) > 1:
            iters = len(fields) ** 2

            for i in range(iters):
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

        else:
            condition_value = [
                {
                    "field": fields[0],
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": "",
                },
                {
                    "field": vector_fields[0],
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": "",
                },
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
        field_types,
    ):
        updated_documents = documents

        for dtype, encoder in encoders.items():
            dtype_fields = [
                field
                for field, field_type in field_types.items()
                if f"_{dtype}_" == field_type
            ]
            updated_documents = encoder.encode_documents(
                dtype_fields,
                updated_documents,
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

    def _get_remaining(self, filters: List[Dict[str, Any]]) -> int:
        return self.get_number_of_documents(
            dataset_id=self.dataset_id,
            filters=filters,
        )

    def _insert_document_vectors(
        self,
        vector_fields: List[str],
        numeric_fields: List[str],
        show_progress_bar: bool,
    ):
        documents = self._get_all_documents(
            dataset_id=self.dataset_id,
            select_fields=vector_fields + numeric_fields,
            show_progress_bar=show_progress_bar,
        )
        schema = self._get_schema()
        if vector_fields:
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
        else:
            vectors = np.array([])

        if numeric_fields:
            numbers = np.array(
                [
                    [
                        document[field] if field in document else 0
                        for field in numeric_fields
                    ]
                    for document in documents
                ]
            )
        else:
            numbers = np.array([])

        if numbers.size > 0 and vectors.size == 0:
            vectors = numbers

        elif numbers.size > 0 and vectors.size > 0:
            vectors = np.hstack([vectors, numbers])

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        vectors = scaler.fit_transform(vectors)
        vectors = np.nan_to_num(vectors)

        if vectors.shape[1] > 512:
            vectors = self._reduce(vectors)

        for index, (document, vector) in enumerate(zip(documents, vectors.tolist())):
            documents[index] = {
                "_id": document["_id"],
                "_vector_": vector,
            }

        self._update_documents(
            dataset_id=self.dataset_id,
            documents=documents,
            show_progress_bar=show_progress_bar,
        )

    @log(fn=log_file)
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
        show_progress_bar: bool = False,
    ) -> None:

        self.dataset_id = dataset_id
        self.schema = self._get_schema()
        self.detailed_schema = self._get_detailed_schema()
        numeric_fields = self._get_numeric_fields()

        if fields:
            if "numeric" in fields:
                if len(fields) == 1:
                    field_types = {}

                else:
                    field_types = self._get_fields(fields)

            else:
                numeric_fields = [field for field in numeric_fields if field in fields]
                field_types = self._get_fields(fields)

        else:
            fields = self.detailed_schema
            field_types = self._get_fields(fields)

        if field_types:
            self._validate_fields(list(field_types))

            self._init_encoders(list(field_types))

            vector_fields = self._get_vector_fields(list(field_types))

            filters = self._get_filters(
                fields=list(field_types),
                vector_fields=vector_fields,
            )

            updating_args = dict(
                encoders=self.encoders,
                field_types=field_types,
            )

            results = self.pull_update_push(
                dataset_id=self.dataset_id,
                update_function=self._encode_documents,
                select_fields=list(fields),
                filters=filters,
                show_progress_bar=show_progress_bar,
                updating_args=updating_args,
            )
            if results["failed_documents"]:
                print(Messages.INSERT_BAD)
                print("There were some errors vectorizing your unstructured data")
        else:
            vector_fields = []

        self._insert_document_vectors(
            vector_fields=vector_fields,
            numeric_fields=numeric_fields,
            show_progress_bar=show_progress_bar,
        )

        new_schema = self._get_schema().keys()

        added_vectors = list(new_schema - self.schema)
        print(Messages.INSERT_GOOD)
        print("The following vector fields were added: " + ", ".join(added_vectors))
