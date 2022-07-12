from relevanceai.constants.errors import MissingPackageError
from relevanceai.operations_new.transform_base import TransformBase

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ModuleNotFoundError:
    raise MissingPackageError("flair", version="0.11.1")

from tqdm.auto import tqdm


class ExtractNounsTransform(TransformBase):
    """
    An operation for extracting noun
    """

    def __init__(
        self,
        fields: list,
        model_name: str,
        output_fields: list,
        cutoff_probability: float,
        stopwords: list = None,
        **kwargs
    ):
        self.fields = fields
        self.model_name = model_name
        self.output_fields = output_fields
        self.cutoff_probability = cutoff_probability
        self.tagger = SequenceTagger.load(model_name)
        from relevanceai.constants import STOPWORDS

        if stopwords is not None:
            self.stopwords = STOPWORDS + stopwords
        else:
            self.stopwords = STOPWORDS

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def process_noun(noun):
        return (
            noun.replace("the ", "")
            .replace("this", "")
            .replace("'s ", "")
            .strip()
            .lower()
        )

    def extract_nouns(self, text, as_documents=False):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        nouns = []
        # Extract the noun phrases
        for entity in sentence.get_spans("np"):
            text = ExtractNounsTransform.process_noun(entity.text.lower().strip())

            if (
                entity.tag == "NP"
                and entity.score >= self.cutoff_probability
                and text not in self.stopwords
            ):
                if as_documents:
                    nouns.append({"noun": text, "score": entity.score})
                else:
                    nouns.append(text)
        return nouns

    def transform(self, docs):
        new_docs = [{"_id": d["_id"]} for d in docs]
        for i, d in enumerate(tqdm(docs)):
            for j, t in enumerate(self.fields):
                value = self.extract_nouns(
                    self.get_field(t, d, missing_treatment="return_empty_string")[:200],
                )
                self.set_field(self.output_fields[j], new_docs[i], value)
        return new_docs

    @property
    def name(self):
        return "extract-nouns"
