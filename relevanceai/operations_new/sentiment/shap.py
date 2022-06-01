from typing import List, Dict, Any, Optional

import numpy as np

from doc_utils import DocUtils

from relevanceai.constants import MissingPackageError

try:
    import shap
except ModuleNotFoundError:
    raise MissingPackageError("shap")

try:
    import transformers

    from transformers.pipelines.base import Pipeline
except ModuleNotFoundError:
    raise MissingPackageError("transformers")

LABEL_MAPPING = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


class SentimentSHAP(DocUtils):
    def __init__(
        self,
        model: str,
        highlight: Optional[bool] = None,
        positive_sentiment_name: Optional[str] = None,
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: Optional[float] = None,
        sentiment_ind: Optional[int] = None,
    ):

        self.sentiment_ind = 2 if sentiment_ind is None else sentiment_ind
        self.positive_sentiment_name = (
            "positive" if positive_sentiment_name is None else positive_sentiment_name
        )
        self.highlight = False if highlight is None else highlight
        self.max_number_of_shap_documents = (
            max_number_of_shap_documents
            if max_number_of_shap_documents is None
            else sentiment_ind
        )
        self.min_abs_score = 0.1 if min_abs_score is None else min_abs_score

        self.classifier = transformers.pipeline(
            return_all_scores=True,
            model=model,
        )

        self.explainer = shap.Explainer(self.classifier)

    @property
    def name(self):
        return type(self.classifier.__class__)

    def get_shap_values(
        self,
        text: str,
    ):
        """Get SHAP values"""
        shap_values = self.explainer([text])
        cohorts = {"": shap_values}
        cohort_exps = list(cohorts.values())
        feature_names = cohort_exps[0].feature_names
        values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
        shap_docs = [
            {"text": v, "score": f}
            for f, v in zip(
                [x[self.sentiment_ind] for x in values[0][0].tolist()], feature_names[0]
            )
        ]
        if self.max_number_of_shap_documents is not None:
            sorted_scores = sorted(shap_docs, key=lambda x: x["score"], reverse=True)
        else:
            sorted_scores = sorted(shap_docs, key=lambda x: x["score"], reverse=True)[
                : self.max_number_of_shap_documents
            ]
        return [d for d in sorted_scores if abs(d["score"]) > self.min_abs_score]

    def analyze_sentiment_with_shap(
        self,
        text: str,
    ):
        labels = self.classifier([text])
        ind_max = np.argmax([l["score"] for l in labels[0]])
        sentiment = labels[0][ind_max]["label"]
        max_score = labels[0][ind_max]["score"]
        sentiment = LABEL_MAPPING.get(sentiment, sentiment)
        if sentiment.lower() == "neutral":
            overall_sentiment = 0
        else:
            overall_sentiment = (
                max_score
                if sentiment.lower() == self.positive_sentiment_name
                else -max_score
            )
        if not self.highlight:
            return {
                "sentiment": sentiment,
                "score": max_score,
                "overall_sentiment": overall_sentiment,
            }
        shap_documents = self.get_shap_values(text)
        return {
            "sentiment": sentiment,
            "score": max_score,
            "overall_sentiment": overall_sentiment,
            "highlight_chunk_": shap_documents,
        }

    def transform_shap(
        self,
        documents: List[Dict[str, Any]],
        text_field: str,
        output_field: str,
    ):
        sentiments = [
            self.analyze_sentiment_with_shap(
                self.get_field(text_field, doc),
            )
            for doc in documents
        ]
        self.set_field_across_documents(
            output_field,
            sentiments,
            documents,
        )
        return documents
