from typing import Dict, Union, Optional, List, Any

from datetime import datetime

import numpy as np

from relevanceai._api import APIClient
from relevanceai.client.helpers import Credentials

from relevanceai.constants import Messages
from relevanceai.constants import IMG_EXTS

from relevanceai.utils.decorators import log

from relevanceai.operations.vector.base import Base2Vec


class VectorizeHelpers(APIClient):
    def __init__(self, log_file, credentials: Credentials):
        self.log_file = log_file
        self.credentials = credentials
        super().__init__(credentials)

    def _get_encoder(self, model: Any) -> Any:
        @log(fn=self.log_file)
        def get_encoder(model, dataset_id, credentials):
            if isinstance(model, str):
                model = model.lower().replace(" ", "").replace("_", "")
                model_name = model
            else:
                model_name = str(model.__class__).split(".")[-1].split("'>")[0]
                return model, model_name

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

                elif model == "onehot":
                    from relevanceai.operations.vector.onehot import OneHot

                    model = OneHot(dataset_id=dataset_id, credentials=credentials)

            else:
                # TODO: this needs to be referenced from relevance.constants.errors
                raise ValueError("ModelNotSupported")

            assert hasattr(model, "vector_length")
            assert model.vector_length is not None

            model.__name__ = model_name

            return model, model_name

        model, model_name = get_encoder(model, self.dataset_id, self.credentials)
        self.model_names.append(model_name.lower())
        return model

    def _reduce(self, vectors: np.ndarray, n_components: int = 512) -> np.ndarray:
        from sklearn.decomposition import PCA

        reducer = PCA(
            n_components=min(vectors.shape[0], vectors.shape[1], n_components)
        )
        reduced = reducer.fit_transform(vectors)
        return reduced

    def _get_models(self, dtype: str) -> List[Any]:

        if hasattr(self, dtype + "encoders"):
            models = getattr(self, dtype + "encoders")

        else:
            models = self._get_default_model(dtype)

        models = [self._get_encoder(model) for model in models]
        return models

    def _get_default_model(self, dtype):

        if dtype == "_text_":
            return ["use"]

        elif dtype == "_image_":
            return ["clip"]

        elif dtype == "_category_":
            return ["onehot"]

        else:
            raise ValueError(
                "We currently do not support a default model for this datatype"
            )


class VectorizeOps(VectorizeHelpers):
    def __init__(
        self,
        credentials: Credentials,
        log_file: str = "vectorize.logs",
        create_feature_vector: bool = False,
        **kwargs,
    ):
        super().__init__(log_file=log_file, credentials=credentials)

        self.feature_vector = create_feature_vector
        self.model_names: List[str] = []

        for encoder_type, encoder in kwargs.items():
            if "_encoders" in encoder_type:
                assert isinstance(encoder, list)
                setattr(self, encoder_type, encoder)

            elif "_encoder" in encoder_type:
                assert not isinstance(encoder, list)
                setattr(self, encoder_type + "s", [encoder])

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def _get_schema(self):
        return self.datasets.schema(self.dataset_id)

    def _get_metadata(self) -> Dict[str, List[str]]:
        return self.datasets.metadata(self.dataset_id)["results"]

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

    def _init_encoders(self):
        for dtype, fields in self.metadata.items():
            if dtype != "_numeric_" and fields:
                models = self._get_models(dtype)
                setattr(self, dtype + "encoders", models)

    def _get_vector_fields(self, fields: List[str]) -> List[str]:
        vector_fields = []

        for dtype, fields in self.metadata.items():
            if dtype != "_numeric_" and fields:
                encoders: List[Base2Vec] = getattr(self, dtype + "encoders")

                new_fields = [
                    encoder.get_default_vector_field_name(field)
                    for encoder in encoders
                    for field in fields
                ]
                vector_fields += new_fields

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
                    {
                        "filter_type": "or",
                        "condition_value": condition_value,
                    }
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
        encoders: Dict[str, List[Base2Vec]],
        field_types: Dict[str, List[str]],
    ):
        updated_documents = documents

        for dtype, vectorizers in encoders.items():
            for vectorizer in vectorizers:
                try:
                    fields = field_types[dtype]
                except:
                    raise ValueError(f"No fields labeled as {dtype}")
                updated_documents = vectorizer.encode_documents(
                    documents=updated_documents,
                    fields=fields,
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
            dataset_id=self.dataset_id,
            metadata=updated_metadata,
        )

    def _insert_document_vectors(
        self,
        fields: List[str],
        show_progress_bar: bool,
    ):
        documents = self._get_all_documents(
            dataset_id=self.dataset_id,
            select_fields=fields,
            show_progress_bar=show_progress_bar,
            include_vector=True,
        )

        vectors = np.concatenate(
            [
                np.concatenate(
                    [np.array(doc[field]).reshape(1, -1) for field in fields],
                    axis=-1,
                )
                for doc in documents
            ],
            axis=0,
        )

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
            documents[index] = {
                "_id": document["_id"],
                document_vector_: vector,
            }

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
        filters: Optional[list] = None,
        **kwargs,
    ) -> None:
        if filters is None:
            filters = []
        self.dataset_id = dataset_id

        self.schema = self._get_schema()
        self.metadata = self._get_metadata()

        if fields:
            for dtype in self.metadata.keys():
                if dtype == "_numeric_" and "_numeric_" in fields:
                    fields += self.metadata[dtype]
                    fields.remove("_numeric_")

                elif dtype == "_category_" and "_category_" in fields:
                    fields += self.metadata[dtype]
                    fields.remove("_category_")

                else:
                    select_fields = []
                    for field in fields:
                        if field in self.metadata[dtype]:
                            select_fields.append(field)
                    self.metadata[dtype] = select_fields
            unstruc_fields = [
                field
                for type in [
                    value for key, value in self.metadata.items() if key != "_numeric_"
                ]
                for field in type
            ]
        else:
            fields = [
                field
                for type in [value for value in self.metadata.values()]
                for field in type
            ]
            unstruc_fields = [
                field
                for type in [
                    value for key, value in self.metadata.items() if key != "_numeric_"
                ]
                for field in type
            ]
            print(
                "No fields were given, vectorizing the following field(s): {}".format(
                    ", ".join(unstruc_fields)
                )
            )

        if unstruc_fields:
            self._validate_fields(fields)

            self._init_encoders()

            vector_fields = self._get_vector_fields(unstruc_fields)
            print(
                "This operation will create the following vector_fields: {}".format(
                    str(vector_fields)
                )
            )

            filters += self._get_filters(
                fields=unstruc_fields,
                vector_fields=vector_fields,
            )

            encoders = {
                encoders: getattr(self, encoders + "encoders")
                for encoders in [
                    var.replace("encoders", "")
                    for var in self.__dict__
                    if "_encoders" in var
                ]
            }

            field_types = {
                dtype: fields
                for dtype, fields in self.metadata.items()
                if dtype != "_numeric_" and fields
            }

            updating_args = dict(
                encoders=encoders,
                field_types=field_types,
            )

            results = self.pull_update_push(
                dataset_id=self.dataset_id,
                update_function=self._encode_documents,
                select_fields=fields,
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

        self._update_vector_metadata(
            metadata=added_vectors,
        )

        if self.feature_vector:
            print(
                "Concatenating the following fields to form a feature vector: {}".format(
                    ", ".join(vector_fields + self.metadata["_numeric_"])
                )
            )
            self._insert_document_vectors(
                fields=vector_fields + self.metadata["_numeric_"],
                show_progress_bar=show_progress_bar,
            )

            new_schema = self._get_schema().keys()
            added_vectors = list(new_schema - self.schema)

        if added_vectors:
            print(Messages.INSERT_GOOD)
            print(
                "The following vector fields were added: " + ", ".join(added_vectors)
            )  # TODO: move messages in Messages class
