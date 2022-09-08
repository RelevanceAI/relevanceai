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
        self.output_fields = output_fields
        self.sensitivity = sensitivity
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

    # def _get_model(self):
    #     try:
    #         import transformers
    #     except ModuleNotFoundError:
    #         print(
    #             "Need to install transformers by running `pip install -q transformers`."
    #         )
    #     self.classifier = transformers.pipeline(
    #         "sentiment-analysis",
    #         return_all_scores=True,
    #         model="cardiffnlp/twitter-roberta-base-sentiment",
    #     )

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
        highlight: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: Optional[int] = None,
        min_abs_score: float = 0.1,
        eps: float = 1e-9,
    ):
        logits = self.classifier(texts, truncation=True, max_length=512)
        scores = pd.DataFrame(
            {
                "negative": self._get_scores(cls="negative", logits=logits),
                "neutral": self._get_scores(cls="neutral", logits=logits),
                "positive": self._get_scores(cls="positive", logits=logits),
            }
        )
        # this is the expected value
        e_x = scores["negative"].values * -1
        e_x += scores["neutral"].values * 0
        e_x += scores["positive"].values * 1

        idxmax = [scores.iloc[index].idxmax(-1) for index in range(len(texts))]
        e_x[e_x == 0] += eps

        if not highlight:
            updates = [
                {
                    "sentiment": idxmax[index],
                    "overall_sentiment_score": e_x[index],
                }
                for index in range(len(texts))
            ]
            return updates

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
            if self.output_fields is not None:
                output_field = self.output_fields[i]
            else:
                output_field = self._get_output_field(t)
            sentiments = self.analyze_sentiment(
                [
                    self.get_field(t, doc, missing_treatment="return_empty_string")
                    for doc in documents
                ],
                highlight=self.highlight,
                max_number_of_shap_documents=self.max_number_of_shap_documents,
                min_abs_score=self.min_abs_score,
            )
            self.set_field_across_documents(output_field, sentiments, sentiment_docs)
        return sentiment_docs

    # def analyze_sentiment(self, text, highlight:bool= True):
    #     try:
    #         from scipy.special import softmax
    #     except ModuleNotFoundError:
    #         print("Need to install scipy")
    #     if not hasattr(self, "model"):
    #         self._get_model()
    #     text = self.preprocess(text)
    #     encoded_input = self.tokenizer(text, return_tensors="pt")
    #     output = self.model(**encoded_input)
    #     scores = output[0][0].detach().numpy()
    #     scores = softmax(scores)
    #     ranking = np.argsort(scores)
    #     ranking = ranking[::-1]
    #     sentiment = self.label_mapping[ranking[0]]
    #     score = np.round(float(scores[ranking[0]]), 4)
    #     return {
    #         "sentiment": sentiment,
    #         "score": np.round(float(scores[ranking[0]]), 4),
    #         "overall_sentiment_score": score if sentiment == "positive" else -score,
    #     }
