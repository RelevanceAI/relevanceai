import itertools
import requests
import warnings
import numpy as np

from typing import List

from relevanceai.utils import DocUtils

BASE_2VEC_DEFINITON = {
    "vector_length": None,
    "description": None,
    "paper": None,
    "repo": None,
    "model_name": None,
    "architecture": None,
    "tasks": None,
    "limitations": None,
    "download_required": None,
    "training_required": None,
    "finetunable": None,
}


class Base2Vec(DocUtils):
    """
    Base class for vector
    """

    def __init__(self):
        self.__dict__.update(BASE_2VEC_DEFINITON)

    @classmethod
    def validate_model_url(cls, model_url: str, list_of_urls: List[str]):
        """
        Validate the model url belongs in the list of urls. This is to help
        users to avoid mis-spelling the name of the model.

        # TODO:
        Improve model URL validation to not include final number in URl string.

        Args:
            model_url: The URl of the the model in question
            list_of_urls: The list of URLS for the model in question

        """
        if model_url in list_of_urls:
            return True

        if "tfhub" in model_url:
            # If the url has a number in it then we can take that into account
            for url in list_of_urls:
                if model_url[:-1] in url:
                    return True
        # TODO: Write documentation link to debugging the Model URL.
        warnings.warn(
            "We have not tested this url. Please use URL at your own risk."
            + "Please use the is_url_working method to test if this is a working url if "
            + "this is not a local directory.",
            UserWarning,
        )

    @staticmethod
    def is_url_working(url):
        response = requests.head(url)
        if response.status_code == 200:
            return True
        return False

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
            >>> ViClient.chunk(documents)
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

    @property
    def __name__(self):
        """
        Return the name of the model. If name is not set, returns the
        model_id.
        """
        if hasattr(self, "_name"):
            return self._name.replace("-", "_")
        elif hasattr(self, "definition"):
            if "/" in self.definition.model_id:
                return self.definition.model_id.split("/")[1].replace("-", "_")
            return self.definition.model_id
        return ""

    @__name__.setter
    def __name__(self, value):
        """
        Set the name.
        """
        setattr(self, "_name", value)

    @property
    def zero_vector(self):
        if hasattr(self, "vector_length"):
            return self.vector_length * [1e-7]
        else:
            raise ValueError("Please set attribute vector_length")

    def is_empty_vector(self, vector):
        return all([x == 1e-7 for x in vector])

    def get_default_vector_field_name(self, field, field_type="vector"):
        if field_type == "vector":
            return field + "_" + self.__name__ + "_vector_"
        elif field_type == "chunkvector":
            return field + "_" + self.__name__ + "_chunkvector_"

    def _encode_document(
        self,
        field,
        doc,
        vector_error_treatment="zero_vector",
        field_type: str = "vector",
    ):
        """Encode document"""
        vector = self.encode(self.get_field(field, doc))
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

    def _encode_chunk_document(
        self,
        chunk_field,
        field,
        doc,
        vector_error_treatment="zero_vector",
        field_type: str = "chunkvector",
    ):
        """Encode a chunk document"""
        chunk_docs = self.get_field(chunk_field, doc)
        if hasattr(self, "bulk_encode"):
            return self.bulk_encode_documents(
                [field],
                chunk_docs,
                field_type=field_type,
                vector_error_treatment=vector_error_treatment,
            )
        elif hasattr(self, "encode"):
            return self.encode_documents(
                [field],
                chunk_docs,
                field_type=field_type,
                vector_error_treatment=vector_error_treatment,
            )

    def _bulk_encode_document(
        self,
        field,
        docs,
        vector_error_treatment: str = "zero_vector",
        field_type="vector",
    ):
        """bulk encode documents"""
        vectors = self.bulk_encode(self.get_field_across_documents(field, docs))
        if vector_error_treatment == "zero_vector":
            self.set_field_across_documents(
                self.get_default_vector_field_name(field, field_type=field_type),
                vectors,
                docs,
            )
            return
        elif vector_error_treatment == "do_not_include":
            [
                self.set_field(
                    self.get_default_vector_field_name(field, field_type=field_type),
                    value=vectors[i],
                    doc=d,
                )
                for i, d in enumerate(docs)
                if not self.is_empty_vector(vectors[i])
            ]
        else:
            [
                self.set_field(
                    self.get_default_vector_field_name(field, field_type=field_type), d
                )
                if not self.is_empty_vector(vectors[i])
                else vector_error_treatment
                for i, d in enumerate(docs)
            ]
            return

    def encode_documents(
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
            field_type:
                Accepts "vector" or "chunkvector"
        """
        for f in fields:
            # Replace with case-switch in future
            [
                self._encode_document(
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
        for f in fields:
            [
                self._encode_chunk_document(
                    chunk_field=chunk_field,
                    field=f,
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
        for f in fields:
            # Replace with case-switch in future
            contained_docs = [d for d in documents if self.is_field(f, d)]
            self._bulk_encode_document(
                f,
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

    def get_result(self, result_text, query_vector):
        return {
            "_search_score": self.get_cosine_similarity(
                query_vector, self.encode(result_text)
            ),
            "text": result_text,
        }

    def explain(
        self,
        query_text,
        result_texts,
        highlight_threshold=0.5,
        max_highlights=None,
        return_cos_similarity_docs: bool = True,
        ignore_warnings: bool = True,
    ):
        import numpy as np
        from IPython.display import display
        from sklearn.preprocessing import MinMaxScaler
        from jsonshower import show_json

        query_vector = self.encode(query_text)
        cos_similarity = list()
        for result_text in result_texts:
            doc = self.get_result(result_text, query_vector)
            doc["explain_chunk"] = [
                {"text": t} for t in self.get_word_combinations(result_text)
            ]
            self.encode_documents(["text"], doc["explain_chunk"])
            for d in doc["explain_chunk"]:
                d["_search_score"] = self.get_cosine_similarity(
                    d[f"text_{self.__name__}_vector_"], query_vector
                )
            doc["explain_chunk"] = sorted(
                doc["explain_chunk"], key=lambda x: x["_search_score"], reverse=True
            )
            cos_similarity.append(doc.copy())

        if return_cos_similarity_docs:
            return cos_similarity

        for r in cos_similarity:
            scaler = MinMaxScaler()
            normed_scores = scaler.fit_transform(
                np.array(
                    [np.exp(_r["_search_score"]) for _r in r["explain_chunk"]]
                ).reshape((-1, 1))
            )
            highlight_colors = [f"rgba(253, 253, 150, {x[0]})" for x in normed_scores]
            text_fields = [
                "explain_chunk." + str(i) + ".text"
                for i in range(len(r["explain_chunk"]))
                if normed_scores[i][0] > 0.5
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if max_highlights is None:
                    display(
                        show_json(
                            [r],
                            highlight_fields={"text": text_fields},
                            highlight_colors=highlight_colors,
                            return_html=False,
                            show_highlight_headers=False,
                            case_insensitive_highlighting=True,
                        )
                    )
                else:
                    display(
                        show_json(
                            [r],
                            highlight_fields={"text": text_fields[:max_highlights]},
                            highlight_colors=highlight_colors,
                            return_html=False,
                            show_highlight_headers=False,
                            case_insensitive_highlighting=True,
                        )
                    )
