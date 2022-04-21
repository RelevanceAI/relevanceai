from typing import Dict, List, Any

import numpy as np

from relevanceai._api import APIClient
from relevanceai.operations.vector.base import Base2Vec


class OneHot(APIClient, Base2Vec):
    def __init__(self, dataset_id, credentials):
        super().__init__(credentials=credentials)

        self._name = "onehot"

        self.dataset_id = dataset_id
        metadata = self.datasets.metadata(dataset_id)["results"]
        self.categorical_fields = metadata["_category_"]
        self.tokens = self._get_tokens(self.categorical_fields)
        self.vector_length = {
            field: len(tokens) for field, tokens in self.tokens.items()
        }

    def _get_tokens(self, categorical_fields) -> Dict[str, Dict[Any, int]]:
        id2token = {
            field: {
                value: index for index, value in enumerate(self._get_u_values(field))
            }
            for field in categorical_fields
        }
        return id2token

    def _get_u_values(self, field):
        agg = self.datasets.aggregate(
            self.dataset_id,
            groupby=[
                {
                    "name": "field",
                    "field": field,
                }
            ],
            metrics=[],
            page_size=10000,
        )["results"]
        u_values = [None] + sorted([field["field"] for field in agg])
        return u_values

    @property
    def __name__(self):
        return self._name

    @__name__.setter
    def __name__(self, value):
        self._name = value

    def _one_hot(self, array, length):
        onehot = np.zeros((array.size, length))
        onehot[np.arange(array.size), array] = 1
        return onehot

    def encode_documents(self, documents, fields):
        tokens = np.array(
            [
                np.array(
                    [
                        self.tokens[field][document[field]] if field in document else 0
                        for field in self.categorical_fields
                    ]
                )
                for document in documents
            ]
        )

        for index in range(tokens.shape[1]):
            field = self.categorical_fields[index]
            onehot = self._one_hot(tokens[:, index], self.vector_length[field]).tolist()

            vector_name = self.get_default_vector_field_name(
                self.categorical_fields[index]
            )
            for document, vector in zip(documents, onehot):
                document[vector_name] = vector

        return documents
