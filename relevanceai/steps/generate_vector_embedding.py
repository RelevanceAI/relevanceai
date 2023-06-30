from pydantic import Field

from typing import Literal, List, Any
from relevanceai.steps.base import Step


class GenerateVectorEmbedding(Step):
    transformation: str = "generate_vector_embedding"
    input: str = Field(...)
    model: Literal[
        "image_text",
        "text_image",
        "all-mpnet-base-v2",
        "clip-vit-b-32-image",
        "clip-vit-b-32-text",
        "clip-vit-l-14-image",
        "clip-vit-l-14-text",
        "sentence-transformers",
        "text-embedding-ada-002",
        "cohere-small",
        "cohere-large",
        "cohere-multilingual-22-12",
    ] = Field(...)

    @property
    def output_spec(self):
        return ["vector"]
