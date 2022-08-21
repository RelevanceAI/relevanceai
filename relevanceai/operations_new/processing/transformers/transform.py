"""
Transformers Pipeline Implementation
"""
from relevanceai.operations_new.transform_base import TransformBase
from typing import Optional


class TransformersPipelineTransform(TransformBase):
    def __init__(
        self, text_fields: list, pipeline, output_fields: Optional[str] = None, **kwargs
    ):
        self.text_fields = text_fields
        self.pipeline = pipeline
        self.task: str = pipeline.task
        self._name: str = pipeline.tokenizer.name_or_path
        self.output_fields = [
            self._generate_output_field(text_field)
            if output_fields is None
            else output_fields
            for text_field in text_fields
        ]
        for k, v in kwargs.items():
            setattr(self, k, v)
        print(f"Output fields are {self.output_fields}")

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return "transformers-pipeline"

    def transform(self, documents):
        for i, text_field in enumerate(self.text_fields):
            texts = self.get_field_across_documents(text_field, documents)
            values = self.pipeline(texts)
            self.set_field_across_documents(self.output_fields[i], values, documents)
        return documents

    def _generate_output_field(self, text_field):
        return f"_{self.task}_.{self.name}." + text_field
