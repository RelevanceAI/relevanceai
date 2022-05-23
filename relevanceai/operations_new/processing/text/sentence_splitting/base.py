from copy import deepcopy
from relevanceai.operations_new.base import OperationBase
from relevanceai.utils import MissingPackageError


class SentenceSplittingBase(OperationBase):
    @property
    def name(self):
        return "sentence_splitting"

    def split_text(self, text, language: str = "en"):
        try:
            from sentence_splitter import split_text_into_sentences
        except ModuleNotFoundError:
            raise MissingPackageError("sentence-splitter")

        return split_text_into_sentences(text=text, language=language)

    def split_text_document(
        self, text_field, document, output_field: str = "_splittextchunk_"
    ):
        """
        Split a text field and store it in other values
        """
        t = self.get_field(text_field, document)
        split_text = self.split_text(t)
        # Format the split text into documents
        split_text_value = [{text_field: s} for s in split_text if s.strip() != ""]
        self.set_field(output_field, document, split_text_value)
        return document

    def transform(
        self,
        text_field,
        documents,
        inplace: bool = True,
        output_field: str = "_splittextchunk_",
    ):
        if not inplace:
            documents = deepcopy(documents)
        # TODO; switch to something faster than list comprehension
        [
            self.split_text_document(
                text_field=text_field, document=document, output_field=output_field
            )
            for document in documents
        ]
        return documents
