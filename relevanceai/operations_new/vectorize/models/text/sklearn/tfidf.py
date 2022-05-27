"""A simple tfidf implementation
"""
from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase


class TfIDFModel(VectorizeModelBase):
    # Simple sklearn tfidf vect
    def __init__(self, text_field: str, output_field: str = None):
        # TODO: refactor this
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vect = TfidfVectorizer()
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
            d[self.output_field] = dense_vectors[i]
        return documents
