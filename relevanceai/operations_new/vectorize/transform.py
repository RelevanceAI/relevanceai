from copy import deepcopy
from abc import abstractmethod

from typing import List, Dict, Any

from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.operations_new.transform_base import TransformBase


class VectorizeTransform(TransformBase):

    models: List[VectorizeModelBase]
    fields: List[str]

    def __init__(
        self,
        fields: List[str],
        models: List[VectorizeModelBase],
        output_fields: list = None,
        **kwargs
    ):
        self.fields = fields
        self.models = [self._get_model(model) for model in models]
        self.vector_fields = []
        for model in self.models:
            for field in self.fields:
                self.vector_fields.append(model.vector_name(field))
        if output_fields is not None:
            if len(output_fields) != len(models) * len(fields):
                raise NotImplementedError(
                    "Output fields only supported for 1 model for now."
                )
        self.output_fields = output_fields

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self):
        return "vectorizing"

    @abstractmethod
    def _get_model(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """It takes a list of documents, and for each document, it runs the document through each of the
        models in the pipeline, and returns the updated documents.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            List[Dict[str, Any]]

        Returns
        -------
            A list of dictionaries.

        """

        updated_documents = deepcopy(documents)

        for model in self.models:
            updated_documents = model.encode_documents(
                documents=updated_documents,
                fields=self.fields,
                output_fields=self.output_fields,
            )

        # removes unnecessary info for updated_where
        updated_documents = [
            {
                key: value
                for key, value in document.items()
                if key not in self.fields or key == "_id"
            }
            for document in updated_documents
        ]

        return updated_documents

    def _get_base_filters(
        self,
    ) -> List:

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

        if len(self.fields) > 1:
            iters = len(self.fields) ** 2

            filters: list = []
            for i in range(iters):
                binary_array = [character for character in str(bin(i))][2:]
                mixed_mask = ["0"] * (
                    len(self.fields) - len(binary_array)
                ) + binary_array
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
                        zip(self.fields, self.vector_fields)
                    )
                ]
                filters += [{"filter_type": "or", "condition_value": condition_value}]

        else:  # Special Case when only 1 field is provided
            condition_value = [
                {
                    "field": self.fields[0],
                    "filter_type": "exists",
                    "condition": "==",
                    "condition_value": " ",
                },
                {
                    "field": self.vector_fields[0],
                    "filter_type": "exists",
                    "condition": "!=",
                    "condition_value": " ",
                },
            ]
            filters = condition_value

        return filters
