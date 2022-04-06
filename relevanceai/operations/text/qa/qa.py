"""
Extract QA to generate summary or contextual question using the T5 model.
"""
from relevanceai.constants.errors import MissingPackageError

try:
    from transformers import pipeline
except:
    raise MissingPackageError("transformers[sentencepiece]")

from relevanceai.operations.base import BaseOps


class QAOps(BaseOps):
    """
    QAOps
    """

    def __init__(self, model_name: str = "mrm8488/deberta-v3-base-finetuned-squadv2"):
        self.model_name = model_name
        self.qa_model = pipeline("question-answering", model=model_name)

    def question_answer(self, question: str, context: str):
        output = self.qa_model(question=question, context=context)
        return {"answer": output["answer"], "score": output["score"]}

    def bulk_question_answer(self, question, contexts):
        return [
            self.qa_model(question=question, context=context) for context in contexts
        ]
