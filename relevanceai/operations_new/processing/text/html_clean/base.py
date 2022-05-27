from copy import deepcopy

from relevanceai.operations_new.base import OperationBase

from relevanceai.operations_new.processing.text.html_clean.helpers import (
    MLStripper,
    BaseTextProcessing,
)


class CleanTextBase(OperationBase, BaseTextProcessing):
    def __init__(self):
        self.stripper = MLStripper()

    @property
    def name(self):
        return "clean_text"

    def clean_text(self, text):
        """
        Clean the text of the individuals
        """
        text = CleanTextBase.normalize_text(
            text, lower=False, remove_punct=False, remove_digit=False
        )
        return self.stripper.clean(text)

    def clean_text_document(
        self, text_field, document, output_field: str = "_cleantext_"
    ):
        """
        Split a text field and store it in other values
        """
        t = self.get_field(text_field, document)
        clean_text = self.clean_text(t)
        # Format the split text into documents
        self.set_field(output_field + "." + text_field, document, clean_text)
        return document

    def transform(
        self,
        text_field,
        documents,
        inplace: bool = True,
        output_field: str = "_cleantext_",
    ):
        if not inplace:
            documents = deepcopy(documents)
        # TODO; switch to something faster than list comprehension
        [
            self.clean_text_document(
                text_field=text_field, document=document, output_field=output_field
            )
            for document in documents
        ]
        return documents
