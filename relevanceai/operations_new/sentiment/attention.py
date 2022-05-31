from typing import List, Dict, Any, Optional

from doc_utils import DocUtils


class SentimentAttention(DocUtils):

    highlight: bool
    min_abs_score: float

    def __init__(self, model):
        self.model = model

    @property
    def name(self):
        return type(self.model.__class__)

    def analyze_sentiment_with_attention(
        self,
        text: str,
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
    ):
        raise NotImplementedError

    def transform_attention(
        self,
        documents: List[Dict[str, Any]],
        text_field: str,
        output_field: str,
    ):
        sentiments = [
            self.analyze_sentiment_with_attention(
                self.get_field(text_field, doc),
                highlight=self.highlight,
                min_abs_score=self.min_abs_score,
            )
            for doc in documents
        ]
        self.set_field_across_documents(output_field, sentiments, documents)
        return documents
