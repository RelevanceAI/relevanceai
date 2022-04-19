"""
Vectorizer model is used to interact with `advanced_vectorize`
"""
from relevanceai.operations.vector import Base2Vec


class Vectorizer(Base2Vec):
    def __init__(
        self,
        field: str,
        alias: str,
        model: callable = None,
        bulk_encode: callable = None,
        encode: callable = None,
    ):
        self.field = field
        self.alias = alias
        self.model = model
        self.bulk_encode = bulk_encode
        self.encode = encode
