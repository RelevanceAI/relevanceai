from typing import Dict, List, Any

from relevanceai.operations_new.vectorize.base import VectorizeBase
from relevanceai.operations_new.vectorize.models.base import ModelBase


class VectorizeTextBase(VectorizeBase):

    TFHUB_MODELS: Dict[str, Dict[str, int]] = {
        "https://tfhub.dev/google/universal-sentence-encoder/4": {
            "vector_length": 512,
        },
    }
    SENTENCE_TRANSFORMER_MODELS: Dict[str, Dict[str, int]] = {
        "distilroberta-base-paraphrase-v1": {
            "vector_length": 768,
        },
        "xlm-r-distilroberta-base-paraphrase-v1": {
            "vector_length": 768,
        },
        "paraphrase-xlm-r-multilingual-v1": {
            "vector_length": 768,
        },
        "distilbert-base-nli-stsb-mean-tokens": {
            "vector_length": 768,
        },
        "bert-large-nli-stsb-mean-tokens": {
            "vector_length": 1024,
        },
        "roberta-base-nli-stsb-mean-tokens": {
            "vector_length": 768,
        },
        "roberta-large-nli-stsb-mean-tokens": {
            "vector_length": 1024,
        },
        "distilbert-base-nli-stsb-quora-ranking": {
            "vector_length": 768,
        },
        "distilbert-multilingual-nli-stsb-quora-ranking": {
            "vector_length": 768,
        },
        "distilroberta-base-msmarco-v1": {
            "vector_length": 768,
        },
        "distiluse-base-multilingual-cased-v2": {
            "vector_length": 512,
        },
        "xlm-r-bert-base-nli-stsb-mean-tokens": {
            "vector_length": 768,
        },
        "bert-base-wikipedia-sections-mean-tokens": {
            "vector_length": 768,
        },
        "LaBSE": {
            "vector_length": 768,
        },
        "average_word_embeddings_glove.6B.300d": {
            "vector_length": 300,
        },
        "average_word_embeddings_komninos": {
            "vector_length": 300,
        },
        "average_word_embeddings_levy_dependency": {
            "vector_length": 768,
        },
        "average_word_embeddings_glove.840B.300d": {
            "vector_length": 300,
        },
        "paraphrase-xlm-r-multilingual-v1": {
            "vector_length": 768,
        },
        "all-MiniLM-L6-v2": {
            "vector_length": 384,
        },
    }

    def _get_model(self, model: Any) -> ModelBase:
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
            if model in self.TFHUB_MODELS:
                from relevanceai.operations_new.vectorize.models.text.tfhub import (
                    TFHubText2Vec,
                )

                vector_length = self.TFHUB_MODELS[model]["vector_length"]

                model = TFHubText2Vec(
                    url=model,
                    vector_length=vector_length,
                )

                return model

            elif model in self.SENTENCE_TRANSFORMER_MODELS:
                from relevanceai.operations_new.vectorize.models.text.sentence_transformers import (
                    SentenceTransformer2Vec,
                )
                from sentence_transformers import SentenceTransformer

                vector_length = self.SENTENCE_TRANSFORMER_MODELS[model]["vector_length"]

                model = SentenceTransformer2Vec(
                    model=SentenceTransformer(model),
                    vector_length=vector_length,
                    model_name=model,
                )

                return model

            else:
                raise ValueError("Model not a valid model string")

        elif isinstance(model, ModelBase):
            return model

        else:
            raise ValueError(
                "Model should either be a supported model string or inherit from ModelBase"
            )
