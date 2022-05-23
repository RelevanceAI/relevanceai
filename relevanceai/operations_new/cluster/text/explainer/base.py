"""Explainer
"""
import numpy as np
import itertools
from typing import List, Callable
from doc_utils import DocUtils


class BaseExplainer(DocUtils):
    """
    Base class for vector
    """

    __name__ = ""

    @classmethod
    def chunk(self, lst: List, chunksize: int):
        """
        Chunk an iterable object in Python but not a pandas DataFrame.
        Args:
            lst:
                Python List
            chunksize:
                The chunk size of an object.
        Example:
            >>> documents = [{...}]
            >>> BaseExplainer.chunk(documents)
        """
        for i in range(0, len(lst), chunksize):
            yield lst[i : i + chunksize]

    def _vector_operation(self, vectors, vector_operation: str = "mean", axis=0):
        """
        Args:
            Vectors: the list of vectors to include
            vector_operation: One of ['mean', 'minus', 'sum', 'min', 'max']
            axis: The axis to which to perform the operation
        """
        if vector_operation == "mean":
            return np.mean(vectors, axis=axis).tolist()
        elif vector_operation == "minus":
            return np.subtract(vectors, axis=axis).tolist()
        elif vector_operation == "sum":
            return np.sum(vectors, axis=axis).tolist()
        elif vector_operation == "min":
            return np.min(vectors, axis=axis).tolist()
        elif vector_operation == "max":
            return np.max(vectors, axis=axis).tolist()
        else:
            return np.mean(vectors, axis=axis).tolist()

    def is_empty_vector(self, vector):
        return all([x == 1e-7 for x in vector])

    def get_default_vector_field_name(self, field, field_type="vector"):
        if field_type == "vector":
            return field + "_" + self.__name__ + "_vector_"
        elif field_type == "chunkvector":
            return field + "_" + self.__name__ + "_chunkvector_"

    def _encode_document(
        self,
        encode_fn,
        field,
        doc,
        vector_error_treatment="zero_vector",
        field_type: str = "vector",
    ):
        """Encode document"""
        vector = encode_fn(self.get_field(field, doc))
        if vector_error_treatment == "zero_vector":
            self.set_field(
                self.get_default_vector_field_name(field, field_type=field_type),
                doc,
                vector,
            )
            return
        elif vector_error_treatment == "do_not_include":
            return
        else:
            if vector is None or self.is_empty_vector(vector):
                vector = vector_error_treatment
            self.set_field(self.get_default_vector_field_name(field), doc, vector)

    def encode_documents(
        self,
        encode_fn,
        fields: list,
        documents: list,
        vector_error_treatment="zero_vector",
        field_type="vector",
    ):
        """
        Encode documents and their specific fields. Note that this runs off the
        default `encode` method. If there is a specific function that you want run, ensure
        that it is set to the encode function.

        Parameters:
            missing_treatment:
                Missing treatment can be one of ["do_not_include", "zero_vector", value].
            documents:
                The documents that are being used
            fields:
                The list of fields to be used
            field_type:
                Accepts "vector" or "chunkvector"
        """
        for f in fields:
            # Replace with case-switch in future
            [
                self._encode_document(
                    encode_fn,
                    f,
                    d,
                    vector_error_treatment=vector_error_treatment,
                    field_type=field_type,
                )
                for d in documents
                if self.is_field(f, d)
            ]
        return documents

    def encode_chunk_documents(
        self,
        chunk_field,
        fields: list,
        documents: list,
        vector_error_treatment: str = "zero_vector",
    ):
        """Encode chunk documents. Loops through every field and then every document.

        Parameters:
            chunk_field: The field for chunking
            fields: A list of fields for chunk documents
            documents: a list of documents
            vector_error_treatment: Vector Error Treatment

        Example:
            >>> chunk_docs = enc.encode_chunk_documents(chunk_field="value", fields=["text"], documents=chunk_docs)

        """
        # Replace with case-switch in future
        for field in fields:
            [
                self._encode_chunk_document(
                    chunk_field=chunk_field,
                    field=field,
                    doc=d,
                    vector_error_treatment=vector_error_treatment,
                    field_type="chunkvector",
                )
                for d in documents
                if self.is_field(chunk_field, d)
            ]
        return documents

    def bulk_encode_documents(
        self,
        fields: list,
        documents: list,
        vector_error_treatment="zero_vector",
        field_type="vector",
    ):
        """
        Encode documents and their specific fields. Note that this runs off the
        default `encode` method. If there is a specific function that you want run, ensure
        that it is set to the encode function.

        Parameters:
            missing_treatment:
                Missing treatment can be one of ["do_not_include", "zero_vector", value].
            documents:
                The documents that are being used
            fields:
                The list of fields to be used
        """
        for field in fields:
            # Replace with case-switch in future
            contained_docs = [d for d in documents if self.is_field(field, d)]
            self._bulk_encode_document(
                field,
                contained_docs,
                vector_error_treatment=vector_error_treatment,
                field_type=field_type,
            )
        return documents

    def get_combinations(self, lst, maximum_span=5):
        for i, j in itertools.combinations(range(len(lst) + 1), 2):
            if j - -i == len(lst):
                continue
            if j - i <= maximum_span:
                yield lst[i:j]

    def get_cosine_similarity(self, vector_1, vector_2):
        from scipy import spatial

        return 1 - spatial.distance.cosine(vector_1, vector_2)

    def get_word_combinations(self, sentence, maximum_span=5):
        for x in list(self.get_combinations(sentence.split(), maximum_span)):
            yield " ".join(x)
        yield ""

    def get_result(self, encode_fn, result_text, query_vector):
        return {
            "_search_score": self.get_cosine_similarity(
                query_vector, encode_fn(result_text)
            ),
            "text": result_text,
        }

    def explain_chunk(
        self,
        encode_fn: Callable,
        query_text,
        result_texts,
        maximum_span: int = 5,
    ):
        query_vector = encode_fn(query_text)
        cos_similarity = list()
        for result_text in result_texts:
            doc = self.get_result(encode_fn, result_text, query_vector)
            doc["explain_chunk"] = [
                {"text": t}
                for t in self.get_word_combinations(
                    result_text, maximum_span=maximum_span
                )
            ]
            self.encode_documents(encode_fn, ["text"], doc["explain_chunk"])
            for d in doc["explain_chunk"]:
                d["_search_score"] = self.get_cosine_similarity(
                    d[f"text_{self.__name__}_vector_"], query_vector
                )
            doc["explain_chunk"] = sorted(
                doc["explain_chunk"], key=lambda x: x["_search_score"], reverse=True
            )
            cos_similarity.append(doc.copy())
        return cos_similarity

    def explain(
        self,
        encode_fn: Callable,
        query_text,
        answer_text,
    ):

        query_vector = encode_fn(query_text)
        return self.explain_from_vector(
            encode_fn=encode_fn, query_vector=query_vector, answer_text=answer_text
        )

    def explain_from_vector(
        self,
        encode_fn: Callable,
        query_vector,
        answer_text,
    ):
        doc = self.get_result(encode_fn, answer_text, query_vector)
        doc["explain_chunk"] = [
            {"text": t} for t in self.get_word_combinations(answer_text)
        ]
        self.encode_documents(encode_fn, ["text"], doc["explain_chunk"])
        for d in doc["explain_chunk"]:
            d["_search_score"] = self.get_cosine_similarity(
                d[f"text_{self.__name__}_vector_"], query_vector
            )
        doc["explain_chunk"] = sorted(
            doc["explain_chunk"], key=lambda x: x["_search_score"], reverse=True
        )
        return doc["explain_chunk"][0]["text"]
