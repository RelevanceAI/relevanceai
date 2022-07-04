"""A simple tfidf implementation
"""
from typing import Optional
from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase


class TfIDFModel(VectorizeModelBase):
    # Simple sklearn tfidf vect
    def __init__(
        self,
        text_field: str,
        output_field: str = None,
        model_kwargs: Optional[dict] = None,
    ):
        # TODO: refactor this
        from sklearn.feature_extraction.text import TfidfVectorizer

        if model_kwargs is None:
            model_kwargs = {}

        if "max_features" not in model_kwargs:
            model_kwargs["max_features"] = 2048

        self.vect = TfidfVectorizer(**model_kwargs)
        self.text_field = text_field
        self.output_field = (
            output_field if output_field is not None else self._get_output_field()
        )

    def _get_output_field(self):
        return self.text_field + "_tfidf_vector_"

    def transform(self, documents):
        text_corpus = self.get_field_across_documents(self.text_field, documents)
        vectors = self.vect.fit_transform(text_corpus)
        dense_vectors = vectors.todense()
        for i, d in enumerate(documents):
            self.set_field(self.output_field, d, dense_vectors[i])
        return documents
