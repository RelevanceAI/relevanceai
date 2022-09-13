"""
Labelling performs a vector search on the labels and fetches the closest
max_number_of_labels.

"""
import re
import itertools
import numpy as np

from typing import Any, Dict, List, Tuple

from sentence_splitter import SentenceSplitter

from relevanceai.operations_new.transform_base import TransformBase


class SurveyTagTransform(TransformBase):
    LABEL_MAPPING = ["contradiction", "neutral", "entailment"]

    def __init__(
        self,
        text_field: str,
        alias: str,
        survey_question: str,
        taxonomy_labels: List[str],
        maximum_tags: int = 5,
    ):

        self.splitter = SentenceSplitter(language="en")
        self.text_field = text_field
        self.survey_question = survey_question
        self.taxonomy_labels = taxonomy_labels
        self.output_field = f"_surveytag_.{text_field}.{alias}"
        self.maximum_tags = maximum_tags

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder("microsoft/deberta-v2-xxlarge-mnli")
        except:
            raise ValueError("Error Loading CrossEncoder Model")

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def create_prediction(topic: str) -> str:
        prediction = f"About {topic}."
        return prediction

    def create_prompt(self, sentence: str, topic: str) -> Tuple[str, str]:
        prompt = (
            self.survey_question + sentence.lower(),
            SurveyTagTransform.create_prediction(topic),
        )
        return prompt

    def split_text(self, text: str) -> List[str]:
        # Add an edge case for text splitting
        split_text = re.split("[?!]", text)
        all_text = itertools.chain.from_iterable(
            [self.splitter.split(s.strip()) for s in split_text]
        )
        return list(all_text)

    def _get_tags(
        self,
        sentence: str,
        minimum_score: float = 0.1,
        verbose: bool = False,
    ):
        sentences = self.split_text(text=sentence)

        entailments = []
        tags: List[Dict[str, Any]] = []
        for sent in sentences:
            labels: List[str] = [
                topic for topic in labels if topic not in [t["label"] for t in tags]
            ]
            if labels == []:
                break
            scores = self.model.predict(
                [
                    self.create_prompt(sent, topic)
                    for i, topic in enumerate(labels)
                    if topic not in [t["label"] for t in tags]
                ]
            )
            entailments = [
                {
                    "prediction": self.LABEL_MAPPING[score_max],
                    "score": SurveyTagTransform.softmax(scores[i])[2],
                }
                for i, score_max in enumerate(scores.argmax(axis=1))
            ]

            for i, doc in enumerate(entailments):
                if doc["prediction"] == "entailment" and doc["score"] > minimum_score:
                    tags.append(
                        {
                            "pred": doc["prediction"],
                            "label": labels[i],
                            "score": doc["score"],
                        }
                    )

            new_tags = [
                t
                for t in tags
                if (t["pred"] == "entailment" and t["score"] > minimum_score)
            ]

        for t in new_tags:
            t.pop("pred")

        if self.maximum_tags is None:
            return new_tags

        if len(new_tags) == 0:
            return [{"label": "[No Tag]"}]

        return new_tags[: self.maximum_tags]

    def _relabel(self, documents: List[Dict[str, Any]]):
        for document in documents:
            text = str(self.get_field(self.text_field, document))
            tags = self._get_tags(text, verbose=False)
            self.set_field(self.output_field, document, tags)
        return documents

    def transform(self, *args, **kwargs):
        return self._relabel(*args, **kwargs)

    @property
    def name(self):
        return "_surveytag_"
