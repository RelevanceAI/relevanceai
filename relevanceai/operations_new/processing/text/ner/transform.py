from copy import deepcopy

from typing import List

from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.utils import MissingPackageError


class ExtractNER(TransformBase):
    def __init__(
        self,
        text_fields: List[str],
        model_id: str = "dslim/bert-base-NER",
        output_field: str = "_ner_",
    ):
        self.text_fields = text_fields
        self.model_id = model_id
        self.output_field = output_field

    @property
    def name(self):
        return "sentence_splitting"

    @property
    def classifier(self):
        if hasattr(self, "_nlp"):
            return self._nlp
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForTokenClassification.from_pretrained(self.model_id)
        self._nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        return self._nlp

    def extract_ner(
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
        new_entities = []
        entities = self.classifier(text)
        # Sample output looks like this
        # [{'end': 33,
        #   'entity': 'B-ORG',
        #   'index': 8,
        #   'score': 0.9665312767028809,
        #   'start': 30,
        #   'word': 'Hot'},
        #  {'end': 39,
        #   'entity': 'I-ORG',
        #   'index': 9,
        #   'score': 0.9847770929336548,
        #   'start': 34,
        #   'word': 'Wheel'},
        # The is_recorded flag allows us to track whether the
        # entity is just 1 syllable and has been recorded or not
        is_recorded = True
        for i, entity in enumerate(entities):
            if entity["entity"].startswith("B-"):
                if is_recorded:
                    word = entity["word"]
                else:
                    new_entities.append(
                        {"word": word, "type": entity["entity"].replace("I-", "")}
                    )

                is_recorded = False
            elif entity["entity"].startswith("I-"):
                if "##" in entity["word"]:
                    word += entity["word"].replace("##", "")
                else:
                    word += " " + entity["word"]

                new_entities.append(
                    {"word": word, "type": entity["entity"].replace("I-", "")}
                )
                is_recorded = True

        return new_entities

    def extract_ner_from_document(
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
            split_text = self.extract_ner(text)

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
            self.extract_ner_from_document(
                document=document,
                text_fields=self.text_fields,
                output_field=self.output_field,
            )
            for document in documents
        ]
        return documents