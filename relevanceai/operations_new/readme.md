# Transform and Operations
- Transform is the step of turning and input into an output.
- Operation will take a Transform, feed it the correct input then take the output and insert it into the right places within Relevance.

# Different base clasess:
1. `transform_base.py`: This is where all the checks and non dataset interactive functions go. <- `transform.py` should inherit this.
2. `ops_run.py`: This is where all the dataset related interactions goes. 
3. `ops_api_base.py`: This is when API client methods are required. <- `ops.py` should inherit this.

# How to create a Operation & Transform
You need two things:
`transform.py`: This will take documents as inputs run the necessary transformation and return updated part of the documents as outputs.
`ops.py`: This will take a transform and run it against a whole dataset.

You can choose to combine the two under `ops.py`, since dataset_ops will always inherit from `ops.py`.

Example of `ops.py`:

```
"""Transformers
"""
from relevanceai.operations_new.processing.transformers.transform import (
    TransformersPipelineTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class TransformersPipelineOps(TransformersPipelineTransform, OperationAPIBase):
    pass
```

Example of `transform.py`:
```
"""
Transformers Pipeline Implementation
"""
from relevanceai.operations_new.transform_base import TransformBase
from typing import Optional


class TransformersPipelineTransform(TransformBase):
    def __init__(self, text_fields: list, pipeline, output_field: Optional[str] = None):
        self.text_fields = text_fields
        self.pipeline = pipeline
        self.task: str = pipeline.task
        self._name: str = pipeline.tokenizer.name_or_path
        self.output_field = (
            self._generate_output_field() if output_field is None else output_field
        )

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return "transformers-pipeline"

    def transform(self, documents):
        for text_field in self.text_fields:
            texts = self.get_field_across_documents(text_field, documents)
            values = self.pipeline(texts)
            self.set_field_across_documents(self.output_field, values, documents)
        return documents

    def _generate_output_field(self):
        return f"_{self.task}_." + self.name + "." + ".".join(self.text_fields)
```