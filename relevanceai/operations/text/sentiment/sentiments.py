"""Add Sentiment to your dataset
"""

# Running a function across each subcluster
import numpy as np
import csv
from urllib.request import urlopen
from relevanceai.constants.errors import MissingPackageError
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

    @property
    def classifier(self):
        if not hasattr(self, "_classifier"):
            import transformers

            self._classifier = transformers.pipeline(
                "sentiment-analysis",
                return_all_scores=True,
                model="cardiffnlp/twitter-roberta-base-sentiment",
            )
        return self._classifier

    def _get_model(self):
        try:
            import transformers
        except ModuleNotFoundError:
            print(
                "Need to install transformers by running `pip install -q transformers`."
            )
        self.classifier = transformers.pipeline(
            "sentiment-analysis",
            return_all_scores=True,
            model="cardiffnlp/twitter-roberta-base-sentiment",
        )

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

    def analyze_sentiment(
        self,
        text,
        include_shap_values: bool = False,
        positive_sentiment_name: str = "positive",
        max_number_of_shap_documents: int = 5,
    ):
        labels = self.classifier([text])
        ind_max = np.argmax([l["score"] for l in labels[0]])
        sentiment = labels[0][ind_max]["label"]
        max_score = labels[0][ind_max]["score"]
        overall_sentiment = (
            max_score if sentiment == positive_sentiment_name else -max_score
        )
        if not include_shap_values:
            return {
                "sentiment": sentiment,
                "score": max_score,
                "overall_sentiment": overall_sentiment,
            }
        shap_documents = self.get_shap_values(
            text,
            sentiment_ind=ind_max,
            max_number_of_shap_documents=max_number_of_shap_documents,
        )
        return {
            "sentiment": sentiment,
            "score": max_score,
            "overall_sentiment": overall_sentiment,
            "shap_chunk_": shap_documents,
        }

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
        self, text: str, sentiment_ind: int = 2, max_number_of_shap_documents: int = 5
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
        return sorted(shap_docs, key=lambda x: x["score"], reverse=True)[
            :max_number_of_shap_documents
        ]

    # def analyze_sentiment(self, text, include_shap_values:bool= True):
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
