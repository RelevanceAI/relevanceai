from typing import List, Dict, Any

from doc_utils import DocUtils

from relevanceai.constants import MissingPackageError

try:
    from transformers import AutoTokenizer

except ModuleNotFoundError:
    raise MissingPackageError("transformers")

from relevanceai.operations_new.sentiment.modules.BERT.ExplanationGenerator import (
    Generator,
)
from relevanceai.operations_new.sentiment.modules.BERT.BertForSequenceClassification import (
    BertForSequenceClassification,
)

CLASSIFICATIONS = ["NEGATIVE", "POSITIVE"]


class SentimentAttention(DocUtils):

    highlight: bool
    min_abs_score: float

    def __init__(
        self,
        model: str = "textattack/bert-base-uncased-SST-2",
    ):
        self.model = BertForSequenceClassification.from_pretrained(model).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # initialize the explanations generator
        self.explainer = Generator(self.model)

    @property
    def name(self):
        return type(self.model.__class__)

    def analyze_sentiment_with_attention(
        self,
        text: str,
    ):
        sentiment = 0
        max_score = 0
        overall_sentiment = 0
        shap_documents = 0

        return {
            "sentiment": sentiment,
            "score": max_score,
            "overall_sentiment": overall_sentiment,
            "highlight_chunk_": shap_documents,
        }

    def transform_attention(
        self,
        documents: List[Dict[str, Any]],
        text_field: str,
        output_field: str,
    ):
        sentiments = [
            self.analyze_sentiment_with_attention(self.get_field(text_field, doc))
            for doc in documents
        ]
        self.set_field_across_documents(
            output_field,
            sentiments,
            documents,
        )
        return documents
