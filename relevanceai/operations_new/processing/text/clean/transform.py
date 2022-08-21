from relevanceai.operations_new.processing.text.clean.helpers import (
    BaseTextProcessing,
    MLStripper,
)

from relevanceai.operations_new.transform_base import TransformBase


class CleanTextTransform(TransformBase):
    def __init__(
        self,
        text_fields: list,
        output_fields: list,
        remove_html_tags: bool = True,
        lower=False,
        remove_punctuation=True,
        remove_digits=True,
        remove_stopwords: list = None,
        lemmatize: bool = False,
        replace_words: dict = None,
        **kwargs
    ):
        if len(text_fields) != len(output_fields):
            raise ValueError("Text fields and output fields are not equal!")
        self.text_fields = text_fields
        self.output_fields = output_fields
        self.remove_html_tags = remove_html_tags
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.replace_words = replace_words
        # Set all the other kwargs!
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clean_text(self, text):
        """
        Clean the text of the individuals
        """
        # TODO: Make this faster by moving the operations into the folder
        try:
            if self.lower:
                text = BaseTextProcessing.lower_text(text)
            if self.remove_punctuation:
                text = BaseTextProcessing.remove_punctuation(text)
            if self.remove_html_tags:
                text = BaseTextProcessing.remove_html_tags(text)
            if self.remove_digits:
                text = BaseTextProcessing.remove_digits(text)
            if self.remove_stopwords:
                text = BaseTextProcessing.remove_stopwords(
                    text, additional_stp_wrds=self.remove_stopwords
                )
            if self.lemmatize:
                text = BaseTextProcessing.lemmatize(text)
            if self.replace_words:
                text = BaseTextProcessing.replace_words(text, self.replace_words)
        except Exception as e:
            import traceback

            traceback.print_exc()
        return text

    def clean_text_document(self, text_field, document, output_field):
        """
        Split a text field and store it in other values
        """
        t = self.get_field(
            text_field, document, missing_treatment="return_empty_string"
        )
        clean_text = self.clean_text(t)
        # Format the split text into documents
        new_doc = {"_id": document["_id"]}
        self.set_field(output_field, new_doc, clean_text)
        return new_doc

    def clean_text_documents(self, text_field, documents, output_field):
        return [
            self.clean_text_document(
                text_field=text_field, document=document, output_field=output_field
            )
            for document in documents
        ]

    def transform(self, documents):
        for i, t in enumerate(self.text_fields):
            new_documents = self.clean_text_documents(
                t, documents, self.output_fields[i]
            )
        return new_documents

    def name(self):
        return 'clean_text'
