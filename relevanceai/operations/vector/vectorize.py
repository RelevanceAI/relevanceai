from typing import Dict, Union, Optional, List, Any

from datetime import datetime

import numpy as np
import warnings

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.constants import Messages
from relevanceai.constants import IMG_EXTS

from relevanceai.utils.decorators import log

from relevanceai.operations.vector.base import Base2Vec


class VectorizeHelpers(APIClient):
    def __init__(self, log_file, credentials: Credentials):
        self.log_file = log_file
        super().__init__(credentials)

    def _get_encoder(self, model: Any) -> Any:
        @log(fn=self.log_file)
        def get_encoder(model):
            if isinstance(model, str):
                model = model.lower().replace(" ", "").replace("_", "")
                model_name = model
            else:
                model_name = str(model.__class__).split(".")[-1].split("'>")[0]
                return model, model_name

            from vectorhub.encoders.text.sentence_transformers import LIST_OF_URLS

            sentence_transformers_model_names = list(LIST_OF_URLS) + [
                "all-MiniLM-L6-v2"
            ]
            sentence_transformers_model_names = {
                name.lower(): name for name in sentence_transformers_model_names
            }

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

                elif model == "clip-image":
                    from vectorhub.bi_encoders.text_image.torch.clip import Clip2Vec

                    model = Clip2Vec()
                    model.encode = model.encode_image

                elif model == "resnet":
                    from vectorhub.encoders.image.fastai import FastAIResnet2Vec

                    model = FastAIResnet2Vec()

                elif model == "mobilenet":
                    from vectorhub.encoders.image.tfhub import MobileNetV12Vec

                    model = MobileNetV12Vec()

                elif model == "clip-text":
                    from vectorhub.bi_encoders.text_image.torch import Clip2Vec

                    model = Clip2Vec()
                    model.encode = model.encode_text

                elif model == "mpnet":
                    from vectorhub.encoders.text.sentence_transformers import (
                        SentenceTransformer2Vec,
                    )

                    model = SentenceTransformer2Vec("all-mpnet-base-v2")
                    model.vector_length = 768

                elif model == "multiqampnet":
                    from vectorhub.encoders.text.sentence_transformers import (
                        SentenceTransformer2Vec,
                    )

                    model = SentenceTransformer2Vec("multi-qa-mpnet-base-dot-v1")
                    model.vector_length = 768

                elif model == "bit":
                    from vectorhub.encoders.image.tfhub import BitMedium2Vec

                    model = BitMedium2Vec()

                elif model in sentence_transformers_model_names:
                    from vectorhub.encoders.text.sentence_transformers import (
                        SentenceTransformer2Vec,
                    )

                    model = SentenceTransformer2Vec(
                        sentence_transformers_model_names[model]
                    )
                    if model.model_name == "all-MiniLM-L6-v2":
                        model.vector_length = 384

                else:
                    raise ValueError("ModelNotSupported")

            else:
                # TODO: this needs to be referenced from relevance.constants.errors
                raise ValueError("ModelNotSupported")

            assert hasattr(model, "vector_length")
            assert model.vector_length is not None

            model.__name__ = model_name

            return model, model_name

        model, model_name = get_encoder(model)
        self.model_names.append(model_name.lower())
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

    def _reduce(self, vectors: np.ndarray, n_components: int = 512) -> np.ndarray:
        from sklearn.decomposition import PCA

        reducer = PCA(
            n_components=min(vectors.shape[0], vectors.shape[1], n_components)
        )
        reduced = reducer.fit_transform(vectors)
        return reduced

    def _get_model_names(self, dtype: str) -> List[Any]:
        if dtype == "_text_":
            if "text" in self.encoders:
                return self.encoders["text"]

            else:
                self.encoders["text"] = ["all-MiniLM-L6-v2"]
                return self.encoders["text"]

        elif dtype == "_image_":
            if "image" in self.encoders:
                return self.encoders["image"]

            else:
                self.encoders["image"] = ["clip"]
                return self.encoders["image"]

        else:
            raise ValueError

    def _get_remaining(self, filters: List[Dict[str, Any]]) -> int:
        return self.get_number_of_documents(dataset_id=self.dataset_id, filters=filters)


class VectorizeOps(VectorizeHelpers):
    def __init__(
        self,
        credentials: Credentials,
        encoders: Optional[Dict[str, List[Any]]] = None,
        log_file: str = "vectorize.logs",
        create_feature_vector: bool = False,
    ):
        super().__init__(log_file=log_file, credentials=credentials)

        self.feature_vector = create_feature_vector
        self.encoders = encoders if encoders is not None else {}
        self.model_names: List[str] = []

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

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

    def _get_numeric_fields(self) -> List[str]:
        numeric_fields = [
            field for field, dtype in self.schema.items() if dtype == "numeric"
        ]
        # facets = self.datasets.facets(self.dataset_id)["results"]
        # stats = {
        #     field: (facets[field]["avg"] - facets[field]["min"])
        #     / (facets[field]["max"] - facets[field]["min"])
        #     for field in [field.replace(" ", "%20") for field in numeric_fields]
        # }
        return numeric_fields

    def _get_fields(self, fields: List[str]) -> Dict[str, str]:
        return {
            field: self.detailed_schema[field]
            for field in fields
            if any(
                dtype in self.detailed_schema[field] for dtype in ["_image_", "_text_"]
            )
        }

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

    def _init_encoders(self, fields: List[str]):
        dtypes = [self.detailed_schema[field] for field in fields]

        for dtype in ["image", "text"]:
            if f"_{dtype}_" in dtypes:
                models = self._get_model_names(dtype=f"_{dtype}_")
                for index, model in enumerate(models):
                    self.encoders[dtype][index] = self._get_encoder(model=model)

    def _get_vector_fields(self, fields: List[str]) -> List[str]:
        vector_fields = []

        for field in fields:
            dtype = self.detailed_schema[field].replace("_", "")
            encoders: List[Base2Vec] = self.encoders[dtype]
            for encoder in encoders:
                vector_field = encoder.get_default_vector_field_name(field)
                vector_fields.append(vector_field)

        return vector_fields

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
                # Creates a binary mask the length of fields provided
                # for two fields, we need 4 iters, going over [(0, 0), (1, 0), (0, 1), (1, 1)]

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
                    {"filter_type": "or", "condition_value": condition_value}
                )

        else:  # Special Case when only 1 field is provided
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
            filters.append({"filter_type": "or", "condition_value": condition_value})

        return filters

    @staticmethod
    def _encode_documents(
        documents, encoders: Dict[str, List[Base2Vec]], field_types: Dict[str, str]
    ):
        updated_documents = documents

        for dtype, vectorizers in encoders.items():
            for vectorizer in vectorizers:
                fields = [
                    field
                    for field, field_type in field_types.items()
                    if f"_{dtype}_" == field_type
                ]
                updated_documents = vectorizer.encode_documents(
                    documents=updated_documents, fields=fields
                )

        return updated_documents

    def _update_vector_metadata(self, metadata: List[str]) -> None:
        updated_metadata = self.datasets.metadata(self.dataset_id)["results"]

        if "_vector_" not in updated_metadata:
            updated_metadata["_vector_"] = {}

        now = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()

        for vector_name in metadata:
            updated_metadata["_vector_"][vector_name] = now

        self.datasets.post_metadata(
            dataset_id=self.dataset_id, metadata=updated_metadata
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

        vectors = scaler.fit_transform(vectors)
        vectors = np.nan_to_num(vectors)

        document_vector_ = f"_dim{vectors.shape[1]}_feature_vector_"
        print(f"Concatenated field is called {document_vector_}")

        for index, (document, vector) in enumerate(zip(documents, vectors.tolist())):
            documents[index] = {"_id": document["_id"], document_vector_: vector}

        self._update_documents(
            dataset_id=self.dataset_id,
            documents=documents,
            show_progress_bar=show_progress_bar,
        )

        self._update_vector_metadata(metadata=[document_vector_])

    def run(
        self,
        dataset_id: str,
        fields: List[str],
        show_progress_bar: bool = True,
        detailed_schema: Optional[Dict[str, Any]] = None,
        filters: Optional[list] = None,
        **kwargs,
    ) -> None:
        if filters is None:
            filters = []
        self.dataset_id = dataset_id
        self.schema = self._get_schema()
        if detailed_schema is not None:
            self.detailed_schema = detailed_schema
        else:
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
            fields_list = list(self.detailed_schema)
            field_types = self._get_fields(fields_list)
            print(
                "No fields were given, vectorizing the following field(s): {}".format(
                    ", ".join(list(field_types))
                )
            )

        if field_types:
            self._validate_fields(list(field_types))

            self._init_encoders(list(field_types))

            vector_fields = self._get_vector_fields(list(field_types))
            print(
                "This operation will create the following vector_fields: {}".format(
                    str(vector_fields)
                )
            )

            filters += self._get_filters(
                fields=list(field_types), vector_fields=vector_fields
            )

            updating_args = dict(encoders=self.encoders, field_types=field_types)

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
                print(
                    "There were some errors vectorizing your unstructured data"
                )  # TODO: move messages in Messages class
                return results
        else:
            vector_fields = []

        new_schema = self._get_schema().keys()
        added_vectors = list(new_schema - self.schema)

        self._update_vector_metadata(metadata=added_vectors)

        if self.feature_vector:
            print(
                "Concatenating the following fields to form a feature vector: {}".format(
                    ", ".join(vector_fields + numeric_fields)
                )
            )
            self._insert_document_vectors(
                vector_fields=vector_fields,
                numeric_fields=numeric_fields,
                show_progress_bar=show_progress_bar,
            )

            new_schema = self._get_schema().keys()
            added_vectors = list(new_schema - self.schema)

        if added_vectors:
            print(Messages.INSERT_GOOD)
            print(
                "The following vector fields were added: " + ", ".join(added_vectors)
            )  # TODO: move messages in Messages class
