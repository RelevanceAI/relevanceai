"""Add Sentiment to your dataset
"""

# Running a function across each subcluster
import csv

from typing import Any, Dict, List, Optional
from urllib.request import urlopen

from relevanceai.operations_new.base import OperationBase

from relevanceai.operations_new.sentiment.shap import SentimentSHAP
from relevanceai.operations_new.sentiment.attention import SentimentAttention


class SentimentBase(OperationBase, SentimentSHAP, SentimentAttention):
    def __init__(
        self,
        text_fields: List[str],
        method: Optional[str] = None,
        model: Optional[str] = None,
        highlight: Optional[bool] = False,
        positive_sentiment_name: Optional[str] = None,
        max_number_of_shap_documents: Optional[int] = None,
        sentiment_ind: Optional[int] = None,
        min_abs_score: Optional[float] = None,
        **kwargs,
    ):
        """
        Sentiment Ops.

        Parameters
        -------------

        model_name: str
            The name of the model

        """
        self.method = "attention" if method is None else method

        if method == "attention":
            model = "textattack/bert-base-uncased-SST-2" if model is None else model
            self.model = SentimentAttention(model=model)

        elif method == "shap":
            model = (
                "siebert/sentiment-roberta-large-english" if model is None else model
            )
            self.model = SentimentSHAP(
                model=model,
                highlight=highlight,
                max_number_of_shap_documents=max_number_of_shap_documents,
                min_abs_score=min_abs_score,
                positive_sentiment_name=positive_sentiment_name,
                sentiment_ind=sentiment_ind,
            )

        self.text_fields = text_fields
        for k, v in kwargs.items():
            setattr(self, k, v)

    def preprocess(self, text: str):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

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
    def name(self):
        return "sentiment"

    def _get_output_field(self, text_field):
        norm_name = self.model.name
        return f"_sentiment_.{text_field}.{norm_name}"

    def transform(
        self,
        documents: List[Dict[str, Any]],
        method: str = "attention",
    ):
        for text_field in self.text_fields:
            output_field = self._get_output_field(text_field)

            if method == "attention":
                documents = self.transform_attention(
                    documents, text_field, output_field
                )

            elif method == "shap":
                # For each document, update the field
                documents = self.transform_shap(documents, text_field, output_field)

            else:
                raise NotImplementedError(
                    "Currently, RelevanceAI does not support sentiment extraction for this method type"
                )

        return documents

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
