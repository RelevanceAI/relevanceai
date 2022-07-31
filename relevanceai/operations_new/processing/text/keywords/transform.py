from relevanceai.operations_new.transform_base import TransformBase
from relevanceai.constants.errors import MissingPackageError


class KeyWordTransform(TransformBase):
    """
    Extract keyphrase from documents
    """

    def __init__(
        self,
        fields: list,
        model_name: str = "all-mpnet-base-v2",
        lower_bound: int = 0,
        upper_bound: int = 3,
        output_fields: list = None,
        stop_words: list = None,
        max_keywords: int = 1,
        use_maxsum: bool = False,
        nr_candidates: int = 20,
        **kwargs
    ):
        self.fields = fields
        self.model_name = model_name
        self.output_fields = output_fields
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stop_words = stop_words
        self.max_keywords = max_keywords

        self.use_maxsum = use_maxsum
        self.nr_candidates = nr_candidates
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_output_field(self, field):
        return "_keyphrase_." + field

    @property
    def name(self):
        return "keyphrase"

    @property
    def keyphrase_model(self):
        if not hasattr(self, "_model"):
            try:
                from keybert import KeyBERT
            except ModuleNotFoundError:
                raise MissingPackageError("keybert")

            self._model = KeyBERT(self.model_name)
        return self._model

    def extract_keyphrase(self, text):
        keywords = self.keyphrase_model.extract_keywords(
            text,
            keyphrase_ngram_range=(self.lower_bound, self.upper_bound),
            stop_words=self.stop_words,
            top_n=self.max_keywords,
            use_maxsum=self.use_maxsum,
            nr_candidates=self.nr_candidates,
        )
        return [{"keyword": k[0], "score": k[1]} for k in keywords[: self.max_keywords]]

    def extract_keyphrases(self, texts):
        return [self.extract_keyphrase(t) for t in texts]

    def transform(self, documents):
        # Extract the keywords from a bunch of documents
        keyphrase_docs = [{"_id": d["_id"]} for d in documents]
        for i, t in enumerate(self.fields):
            if self.output_fields is not None:
                output_field = self.output_fields[i]
            else:
                output_field = self._get_output_field(t)
            texts = self.get_field_across_documents(t, documents)
            keyphrases = self.extract_keyphrases(texts)
            self.set_field_across_documents(output_field, keyphrases, keyphrase_docs)
        return keyphrase_docs
