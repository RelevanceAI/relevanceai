"""Add Sentiment to your dataset
"""

# Running a function across each subcluster
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
except ModuleNotFoundError:
    print("Need to install transformers")


class TextSentiment:
    def preprocess(self, text: str):
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def get_model(self, model="cardiffnlp/twitter-roberta-base-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

    def get_label_mapping(self, task: str):
        labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        labels = [row[1] for row in csvreader if len(row) > 1]
        return labels

    @property
    def label_mapping(self):
        return {0: "negative", 1: "neutral", 2: "positive"}

    def analyze_sentiment(self, text):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        return {
            "sentiment": self.label_mapping[ranking[0]],
            "score": np.round(float(scores[ranking[0]]), 4),
        }


# sentiment = TextSentiment()
# sentiment.get_model()
