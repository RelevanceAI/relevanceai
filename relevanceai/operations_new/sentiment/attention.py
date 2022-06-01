from typing import List, Dict, Any, Optional

from doc_utils import DocUtils

from relevanceai.constants import MissingPackageError

try:
    import torch
except ModuleNotFoundError:
    raise MissingPackageError("torch")

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
    def __init__(
        self,
        model: str = "textattack/bert-base-uncased-SST-2",
        threshold: Optional[float] = None,
    ):
        self.threshold = 0.1 if threshold is None else threshold
        self.model_name = model
        self.model = BertForSequenceClassification.from_pretrained(model).to("cuda")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # initialize the explanations generator
        self.explainer = Generator(self.model)

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            self.device = "cpu"

    @property
    def name(self):
        try:
            return self.model_name.split("/")[-1]
        except:
            return self.model_name

    def analyze_sentiment(
        self,
        texts: List[str],
        text_field: str,
    ):
        encoding = self.tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # generate an explanation for the input
        expl = self.explainer.generate_full_lrp(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # normalize scores
        min = expl.min(dim=-1).values.unsqueeze(-1)
        max = expl.max(dim=-1).values.unsqueeze(-1)
        expl = (expl - min) / (max - min)

        # get the model classification
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ].softmax(
            dim=-1,
        )

        output = 2 * output[:, 0] - 1
        expl = expl[:, 1:].tolist()

        # if the classification is negative, higher explanation scores are more negative
        # flip for visualization

        sentiment = []
        for score, attention, text in zip(output, expl, texts):
            score = score.item()

            sign = 1
            if score < -self.threshold:
                class_name = "Negative"
                sign = -1

            elif score > self.threshold:
                class_name = "Positive"

            else:
                class_name = "Neutral"

            words = text.split(" ")
            explain_chunk = [
                {text_field: word, "value": sign * value}
                for word, value in zip(words, attention[: len(words)])
            ]

            sentiment.append(
                {
                    self.name: dict(
                        sentiment=class_name,
                        overall_sentiment=score,
                        explain_chunk_=explain_chunk,
                    )
                }
            )

        return sentiment

    def transform(
        self,
        documents: List[Dict[str, Any]],
        text_field: str,
        output_field: str,
    ):
        sentiments = self.analyze_sentiment(
            [self.get_field(text_field, document) for document in documents],
            text_field,
        )
        self.set_field_across_documents(
            output_field,
            sentiments,
            documents,
        )
        return documents
