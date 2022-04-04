"""Add Sentiment to your dataset
"""

# Running a function across each subcluster
import numpy as np
import csv
from urllib.request import urlopen
from relevanceai.operations.base import BaseOps


class SentimentOps(BaseOps):
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Sentiment Ops.

        Parameters
        -------------

        model_name: str
            The name of the model

        """
        self.model_name = model_name

    def preprocess(self, text: str):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def _get_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ModuleNotFoundError:
            print(
                "Need to install transformers by running `pip install -q transformers`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def _get_label_mapping(self, task: str):

        labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        labels = [row[1] for row in csvreader if len(row) > 1]
        return labels

    @property
    def label_mapping(self):
        return {0: "negative", 1: "neutral", 2: "positive"}

    def analyze_sentiment(self, text):
        try:
            from scipy.special import softmax
        except ModuleNotFoundError:
            print("Need to install scipy")
        if not hasattr(self, "model"):
            self._get_model()
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        sentiment = self.label_mapping[ranking[0]]
        score = np.round(float(scores[ranking[0]]), 4)
        return {
            "sentiment": sentiment,
            "score": np.round(float(scores[ranking[0]]), 4),
            "overall_sentiment_score": score if sentiment == "positive" else -score,
        }
