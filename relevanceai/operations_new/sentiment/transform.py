"""Add Sentiment to your dataset
"""

# Running a function across each subcluster
import pandas as pd
import numpy as np
import csv
from typing import Dict, List, Optional
from urllib.request import urlopen
from relevanceai.constants.errors import MissingPackageError
from relevanceai.operations_new.transform_base import TransformBase


class SentimentTransform(TransformBase):
    def __init__(
        self,
        text_fields: list,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
        output_fields: list = None,
        sensitivity: float = 0,
        device: int = None,
        strategy: str = "value_max",
        eps: float = 1e-9,
        **kwargs,
    ):
        """
        Sentiment Ops.

        Parameters
        -------------

        model_name: str
            The name of the model
        sensitivity: float
            How confident it is about being `neutral`. If you are dealing with news sources,
            you probably want less sensitivity

        """
        self.model_name = model_name
        self.text_fields = text_fields
        self.highlight = highlight
        self.positive_sentiment_name = positive_sentiment_name
        self.max_number_of_shap_documents = max_number_of_shap_documents
        self.min_abs_score = min_abs_score
        self.output_fields = (
            output_fields
            if output_fields is not None
            else [self._get_output_field(t) for t in text_fields]
        )
        self.sensitivity = sensitivity
        self.strategy = strategy
        self.eps = eps
        self.device = self.get_transformers_device(device)

        import transformers

        # Tries to set it to set it on GPU
        self._classifier = transformers.pipeline(
            return_all_scores=True, model=self.model_name, device=self.device
        )

        for k, v in kwargs.items():
            setattr(self, k, v)

    def preprocess(self, text: str):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    @property
    def classifier(self):
        return self._classifier

    def _get_label_mapping(self, task: str):
        # Note: this is specific to the current model
        labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        labels = [row[1] for row in csvreader if len(row) > 1]
        return labels

    @property
    def label_mapping(self):
        return {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    def _get_scores(
        self, cls: str, logits: List[List[Dict[str, float]]]
    ) -> List[float]:
        return [
            list(filter(lambda label: label["label"] == cls, logit))[0]["score"]
            for logit in logits
        ]

    def analyze_sentiment(
        self,
        texts: List[str],
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
    ):
        logits = self.classifier(texts, truncation=True, max_length=512)
        scores = pd.DataFrame(
            {
                "negative": self._get_scores(cls="LABEL_0", logits=logits),
                "neutral": self._get_scores(cls="LABEL_1", logits=logits),
                "positive": self._get_scores(cls="LABEL_2", logits=logits),
            }
        )

        if not self.highlight:
            if self.strategy == "expected_value":
                scores, labels = self._expected_value_scoring(
                    scores=scores,
                    texts=texts,
                )

            elif self.strategy == "value_max":
                scores, labels = self._value_max_scoring(
                    scores=scores,
                    texts=texts,
                )

            else:
                raise ValueError

            updates = [
                {
                    "sentiment": scores[index],
                    "overall_sentiment_score": labels[index],
                }
                for index in range(len(texts))
            ]
            return updates

        else:
            return self._shap_documents(
                scores=scores,
                texts=texts,
                max_number_of_shap_documents=max_number_of_shap_documents,
                min_abs_score=min_abs_score,
            )

    def _get_idxmax(self, scores: pd.DataFrame, texts: List[str]):
        return [scores.iloc[index].idxmax(-1) for index in range(len(texts))]

    def _get_value_max_scores(self, scores: pd.DataFrame):
        negative_mask = scores["negative"].values > scores["positive"].values
        positive_mask = scores["negative"].values <= scores["positive"].values
        a_max = (
            -scores["negative"].values * negative_mask
            + scores["positive"].values * positive_mask
        )
        a_max[a_max == 0] += self.eps
        return a_max

    def _value_max_scoring(
        self,
        scores: pd.DataFrame,
        texts: List[str],
    ):
        a_max = self._get_value_max_scores(scores=scores)
        idxmax = self._get_idxmax(scores=scores, texts=texts)
        return a_max, idxmax

    def _get_expected_value_score(self, scores: pd.DataFrame):
        # this is the expected value
        e_x = scores["negative"].values * -1
        e_x += scores["positive"].values
        e_x[e_x == 0] += self.eps
        return e_x

    def _expected_value_scoring(
        self,
        scores: pd.DataFrame,
        texts: List[str],
    ):
        e_x = self._get_expected_value_score(scores=scores)
        idxmax = self._get_idxmax(scores=scores, texts=texts)
        return e_x, idxmax

    def _shap_documents(
        self,
        scores: pd.DataFrame,
        texts,
        max_number_of_shap_documents: int = None,
        min_abs_score: float = 0.1,
    ):
        idxmax = self._get_idxmax(scores=scores, texts=texts)
        shap_documents = [
            self.get_shap_values(
                texts[index],
                sentiment_ind=idxmax[index],
                max_number_of_shap_documents=max_number_of_shap_documents,
                min_abs_score=min_abs_score,
            )
            for index in range(len(texts))
        ]

        max_scores = scores.values[np.argmax(scores.values, axis=-1)]
        e_x = self._get_expected_value_score(scores=scores)

        updates = [
            {
                "sentiment": idxmax[index],
                "score": max_scores[index],
                "overall_sentiment": e_x[index],
                "highlight_chunk_": shap_documents[index],
            }
            for index in range(len(texts))
        ]
        return updates

    @property
    def explainer(self):
        if hasattr(self, "_explainer"):
            return self._explainer
        else:
            try:
                import shap
            except ModuleNotFoundError:
                raise MissingPackageError("shap")
            self._explainer = shap.Explainer(self.classifier)
            return self._explainer

    def _calculate_overall_sentiment(self, score: float, label: str) -> float:
        if label.lower().strip() == "positive":
            return score
        else:
            return -score

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

    @property
    def name(self):
        return "sentiment"

    def _get_output_field(self, text_field):
        norm_name = self.model_name.replace("/", "-")
        return f"_sentiment_.{text_field}.{norm_name}"

    def transform(self, documents):
        # For each document, update the field
        sentiment_docs = [{"_id": d["_id"]} for d in documents]
        for i, t in enumerate(self.text_fields):
            sentiments = self.analyze_sentiment(
                [
                    self.get_field(t, doc, missing_treatment="return_empty_string")
                    for doc in documents
                ],
                max_number_of_shap_documents=self.max_number_of_shap_documents,
                min_abs_score=self.min_abs_score,
            )
            output_field = self.output_fields[i]
            self.set_field_across_documents(output_field, sentiments, sentiment_docs)
        return sentiment_docs
