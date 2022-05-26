from typing import Any

from relevanceai.operations_new.vectorize.base import VectorizeBase
from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.operations_new.vectorize.models.text.mappings import *


class VectorizeTextBase(VectorizeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_model(self, model: Any) -> VectorizeModelBase:
        """If the model is a string, then it is either a TFHUB model or a Sentence Transformer model. If it
        is a TFHUB model, then return None. If it is a Sentence Transformer model, then return a
        SentenceTransformer2Vec object

        Parameters
        ----------
        model : Any
            str = "bert-base-nli-mean-tokens"

        Returns
        -------
            The model is being returned.

        """
        if isinstance(model, str):
            if model in TFHUB_MODELS:
                from relevanceai.operations_new.vectorize.models.text.tfhub import (
                    TFHubText2Vec,
                )

                vector_length = TFHUB_MODELS[model]["vector_length"]

                model = TFHubText2Vec(
                    url=model,
                    vector_length=vector_length,
                )

                return model

            elif model in SENTENCE_TRANSFORMER_MODELS:
                from relevanceai.operations_new.vectorize.models.text.sentence_transformers import (
                    SentenceTransformer2Vec,
                )
                from sentence_transformers import SentenceTransformer

                vector_length = SENTENCE_TRANSFORMER_MODELS[model]["vector_length"]

                model = SentenceTransformer2Vec(
                    model=SentenceTransformer(model),
                    vector_length=vector_length,
                    model_name=model,
                )

                return model

            else:  # assume model is sentence transformer model
                from relevanceai.operations_new.vectorize.models.text.sentence_transformers import (
                    SentenceTransformer2Vec,
                )
                from sentence_transformers import SentenceTransformer

                vector_length = None

                model = SentenceTransformer2Vec(
                    model=SentenceTransformer(model),
                    vector_length=vector_length,
                    model_name=model,
                )

                return model

        elif isinstance(model, VectorizeModelBase):
            return model

        else:
            raise ValueError(
                "Model should either be a supported model string or inherit from ModelBase"
            )
