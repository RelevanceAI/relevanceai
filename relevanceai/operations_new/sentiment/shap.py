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

LABEL_MAPPING = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


class SentimentSHAP(DocUtils):

    highlight: bool
    max_number_of_shap_documents: Optional[int]
    min_abs_score: float
    classifier: Pipeline

    def __init__(self, model: str):

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
        sentiment_ind: int = 2,
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
    ):
        """Get SHAP values"""
        shap_values = self.explainer([text])
        cohorts = {"": shap_values}
        cohort_labels = list(cohorts.keys())
        cohort_exps = list(cohorts.values())
        features = cohort_exps[0].data
        feature_names = cohort_exps[0].feature_names
        values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
        shap_docs = [
            {"text": v, "score": f}
            for f, v in zip(
                [x[sentiment_ind] for x in values[0][0].tolist()], feature_names[0]
            )
        ]
        if max_number_of_shap_documents is not None:
            sorted_scores = sorted(shap_docs, key=lambda x: x["score"], reverse=True)
        else:
            sorted_scores = sorted(shap_docs, key=lambda x: x["score"], reverse=True)[
                :max_number_of_shap_documents
            ]
        return [d for d in sorted_scores if abs(d["score"]) > min_abs_score]

    def analyze_sentiment_with_shap(
        self,
        text: str,
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
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
                if sentiment.lower() == positive_sentiment_name
                else -max_score
            )
        if not highlight:
            return {
                "sentiment": sentiment,
                "score": max_score,
                "overall_sentiment": overall_sentiment,
            }
        shap_documents = self.get_shap_values(
            text,
            sentiment_ind=ind_max,
            max_number_of_shap_documents=max_number_of_shap_documents,
            min_abs_score=min_abs_score,
        )
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
                highlight=self.highlight,
                max_number_of_shap_documents=self.max_number_of_shap_documents,
                min_abs_score=self.min_abs_score,
            )
            for doc in documents
        ]
        self.set_field_across_documents(output_field, sentiments, documents)
        return documents
