import copy
import warnings

from typing import Any, Dict, List, Union

from relevanceai.utils import DocUtils


class DocumentVectorizer(DocUtils):
    def __init__(
        self,
        model_name: str = "gtr-t5-large",
        show_progress_bar: bool = True,
        batch_size: int = 32,
    ) -> None:
        """
        A wrapper class for Sentence Transformers with local RelevanceAI SDK integrations
        """

        self.model_name = f"sentence-transformers/{model_name}"
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.encode_kwargs = dict(
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "please `pip install sentence-transformers` to use DocumentVectorizer"
            )

        try:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encode_kwargs["device"] = self.device
        except ModuleNotFoundError:
            self.device = None
            warnings.warn("please `pip install torch` to use DocumentVectorizer")

        print(f"Device selected: {self.device}")

    def encode(
        self,
        document: Union[Dict[str, Any], List[Dict[str, Any]]],
        field: str,
        inplace: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """support single/multi documents with one method"""

        if isinstance(document, dict):
            return self._encode(document=document, field=field, inplace=inplace)

        elif isinstance(document, list):
            return self._bulk_encode(documents=document, field=field, inplace=inplace)

        else:
            raise ValueError(
                "can only vectorizer single documents or lists of documents"
            )

    def _get_vector_field(self, path: str) -> str:
        return f"{path}_vector_"

    def _encode(
        self,
        document: Dict[str, Any],
        field: str,
        inplace: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Vectorize a single document
        """
        if not inplace:
            new_document = copy.copy(document)
        else:
            new_document = document

        value = self.get_field(field, new_document)

        vector = self.model.encode(value, **self.encode_kwargs)
        vector = vector.tolist()

        vector_field = self._get_vector_field(field)
        self.set_field(
            field=vector_field,
            doc=new_document,
            value=vector,
        )
        return new_document

    def _bulk_encode(
        self,
        documents: List[Dict[str, Any]],
        field: str,
        inplace: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Bulk vectorize documents locally
        """
        if not inplace:
            new_documents = copy.copy(documents)
        else:
            new_documents = documents

        values = self.get_field_across_documents(field=field, docs=new_documents)

        vectors = self.model.encode(values, **self.encode_kwargs)
        vectors = vectors.tolist()

        vector_field = self._get_vector_field(field)
        self.set_field_across_documents(
            field=vector_field,
            docs=new_documents,
            values=vectors,
        )
        return new_documents
