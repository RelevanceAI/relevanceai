from copy import deepcopy

from typing import List

from relevanceai.operations_new.base import OperationBase
from relevanceai.utils import MissingPackageError


class SentenceSplittingBase(OperationBase):
    def __init__(
        self,
        text_fields: List[str],
        language: str = "en",
        inplace: bool = True,
        output_field: str = "_splittextchunk_",
    ):
        self.text_fields = text_fields
        self.language = language
        self.inplace = inplace
        self.output_field = output_field

    @property
    def name(self):
        return "sentence_splitting"

    def split_text(
        self,
        text,
    ):
        """The function takes a text and a language as input and returns a list of sentences

        Parameters
        ----------
        text
            The text to split into sentences.
        language : str, optional
            The language of the text.

        Returns
        -------
            A list of sentences.

        """
        try:
            from sentence_splitter import split_text_into_sentences
        except ModuleNotFoundError:
            raise MissingPackageError("sentence-splitter")

        return split_text_into_sentences(text=text, language=self.language)

    def split_text_document(
        self,
        document,
    ):
        """It takes a document, splits it into chunks, and then returns a list of documents, each
        containing a single chunk

        Parameters
        ----------
        text_fields
            The fields in the document that contain the text to be split.
        document
            The document to be split.
        output_field : str, optional
            The name of the field that will contain the split text.

        Returns
        -------
            A list of dictionaries.

        """
        for text_field in self.text_fields:
            text = self.get_fields(text_field, document)
            split_text = self.split_text(text)

            # Format the split text into documents
            split_text_value = [{text_field: s} for s in split_text if s.strip() != ""]
            self.set_field(
                self.output_field,
                document,
                split_text_value,
            )

        return document

    def transform(
        self,
        documents,
    ):
        """It takes a list of documents, and for each document, it splits the text into chunks and adds the
        chunks to the document

        Parameters
        ----------
        text_fields
            a list of fields in the document that contain text to be split
        documents
            list of documents to be split
        inplace : bool, optional
            bool = True
        output_field : str, optional
            str = "_splittextchunk_"

        Returns
        -------
            A list of documents

        """
        if not self.inplace:
            documents = deepcopy(documents)

        # TODO; switch to something faster than list comprehension
        [
            self.split_text_document(
                document=document,
                text_fields=self.text_fields,
                output_field=self.output_field,
            )
            for document in documents
        ]
        return documents
