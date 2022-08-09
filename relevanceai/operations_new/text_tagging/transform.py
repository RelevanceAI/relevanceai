"""
Labelling performs a vector search on the labels and fetches the closest
max_number_of_labels.

"""
from copy import deepcopy
from tkinter import N
from typing import Any, Dict, List

from numpy import maximum

from relevanceai.operations_new.transform_base import TransformBase


class TextTagTransform(TransformBase):
    def __init__(
        self,
        text_field: str,
        labels: list,
        minimum_score: float = 0.25,
        model_id=None,
        maximum_number_of_labels: int = 5,
        **kwargs,
    ):
        self.text_field = text_field
        self.labels = labels
        self.model_id = model_id
        self.minimum_score = minimum_score
        self.maximum_number_of_labels = maximum_number_of_labels
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def classifier(self):
        if not hasattr(self, "_classifier"):
            from transformers import pipeline

            try:
                self._classifier = pipeline(
                    "zero-shot-classification", model=self.model_id, device=0
                )
            except:
                self._classifier = pipeline(
                    "zero-shot-classification", model=self.model_id, device=-1
                )
        return self._classifier

    def transform(  # type: ignore
        self,
        documents,
    ) -> List[Dict[str, Any]]:
        new_docs = []
        for doc in documents:
            query = self.get_field(self.text_field, doc)
            labels = self.tag_text(query, self.labels)
            new_doc = {"_id": doc["_id"]}
            self.set_field(
                self._generate_output_field(self.text_field), new_doc, labels
            )
            new_docs.append(new_doc)

        return new_docs

    @property
    def name(self):
        return "texttag"

    def tag_text(
        self,
        query,
        labels,
    ):
        """It takes a query vector, a vector field, a list of documents, and a few other parameters, and
        returns a list of documents sorted by their cosine similarity to the query vector

        Parameters
        ----------
        query_vector
            the vector you want to compare against
        vector_field
            the field in the documents that contains the vector
        documents
            list of documents
        reverse, optional
            True/False
        score_field : str, optional
            str = "_label_score"
        max_number_of_labels : int, optional
            int = 1,
        similarity_threshold : float, optional
            float = 0,

        Returns
        -------
            A list of dictionaries.

        """
        output = self.classifier(query, labels, multi_label=True)
        labels = [
            l
            for i, l in enumerate(output["labels"])
            if output["scores"][i] > self.minimum_score
        ]

        counter = 0
        new_labels = []
        for i, label in enumerate(labels):
            if (
                self.maximum_number_of_labels is not None
                and counter >= self.maximum_number_of_labels
            ):
                return new_labels

            if label not in new_labels:
                new_labels.append(label)
                counter += 1

        return new_labels

    def get_operation_metadata(self) -> Dict[str, Any]:
        return dict(
            operation="texttags",
            values=str(
                {
                    "minimum_score": self.minimum_score,
                    "max_number_of_labels": self.maximum_number_of_labels,
                }
            ),
        )

    def _generate_output_field(self, field):
        return f"_{self.name}_.{field.lower().replace(' ', '_')}.{self.model_id}"
