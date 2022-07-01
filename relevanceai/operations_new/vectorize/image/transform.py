from typing import Dict, List, Any

from relevanceai.operations_new.vectorize.transform import VectorizeTransform
from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.operations_new.vectorize.models.image.mappings import *


class VectorizeImageTransform(VectorizeTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_model(self, model: Any) -> VectorizeModelBase:
        """If the model is a string, then check if it's in the TFHUB_MODELS dictionary. If it is, then
        return a TFHubImage2Vec object. If it's not, then raise a ValueError

        Parameters
        ----------
        model : Any
            Any

        Returns
        -------
            The model is being returned.

        """

        if isinstance(model, str):
            if model in TFHUB_MODELS:
                from relevanceai.operations_new.vectorize.models.image.tfhub import (
                    TFHubImage2Vec,
                )

                vector_length = TFHUB_MODELS[model]["vector_length"]
                url = TFHUB_MODELS[model]["url"]

                image_dimensions = (
                    TFHUB_MODELS[model]["image_dimensions"]
                    if "image_dimensions" in TFHUB_MODELS[model]
                    else None
                )
                model = TFHubImage2Vec(
                    url=url,
                    vector_length=vector_length,
                    image_dimensions=image_dimensions,
                )

                return model

            if model in CLIP_MODELS:
                from relevanceai.operations_new.vectorize.models.image.clip import (
                    ClipImage2Vec,
                )

                vector_length = CLIP_MODELS[model]["vector_length"]
                url = CLIP_MODELS[model]["url"]

                model = ClipImage2Vec(
                    url=url,
                    vector_length=vector_length,
                )

                return model

            else:
                raise ValueError("Model not a valid model string")

        elif isinstance(model, VectorizeModelBase):
            return model

        else:
            raise ValueError(
                "Model should either be a supported model string or inherit from ModelBase"
            )
