from typing import Dict, List, Any

from relevanceai.operations_new.vectorize.base import VectorizeBase
from relevanceai.operations_new.vectorize.models.base import ModelBase


class VectorizeImageBase(VectorizeBase):

    CLIP_MODELS: Dict[str, Dict[str, int]] = {
        "clip": {
            "vector_length": 512,
        }
    }

    TFHUB_MODELS: Dict[str, Dict[str, int]] = {
        "https://tfhub.dev/google/bit/s-r50x1/1": {
            "vector_length": 2048,
        },
        "https://tfhub.dev/google/bit/s-r50x3/1": {
            "vector_length": 6144,
        },
        "https://tfhub.dev/google/bit/s-r101x1/1": {
            "vector_length": 2048,
        },
        "https://tfhub.dev/google/bit/s-r101x3/1": {
            "vector_length": 6144,
        },
        "https://tfhub.dev/google/bit/s-r152x4/1": {
            "vector_length": 8192,
        },
        "https://tfhub.dev/google/bit/m-r50x1/1": {
            "vector_length": 2048,
        },
        "https://tfhub.dev/google/bit/m-r50x3/1": {
            "vector_length": 6144,
        },
        "https://tfhub.dev/google/bit/m-r101x1/1": {
            "vector_length": 2048,
        },
        "https://tfhub.dev/google/bit/m-r101x3/1": {
            "vector_length": 6144,
        },
        "https://tfhub.dev/google/bit/m-r152x4/1": {
            "vector_length": 8192,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/4": {
            "vector_length": 1024,
            "image_dimensions": 224,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/4": {
            "vector_length": 1024,
            "image_dimensions": 192,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/4": {
            "vector_length": 1024,
            "image_dimensions": 160,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/4": {
            "vector_length": 1024,
            "image_dimensions": 128,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/4": {
            "vector_length": 768,
            "image_dimensions": 224,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/4": {
            "vector_length": 768,
            "image_dimensions": 192,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/4": {
            "vector_length": 768,
            "image_dimensions": 160,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/4": {
            "vector_length": 768,
            "image_dimensions": 128,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/4": {
            "vector_length": 512,
            "image_dimensions": 224,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/4": {
            "vector_length": 512,
            "image_dimensions": 192,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/4": {
            "vector_length": 512,
            "image_dimensions": 160,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/4": {
            "vector_length": 512,
            "image_dimensions": 128,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/4": {
            "vector_length": 256,
            "image_dimensions": 224,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/4": {
            "vector_length": 256,
            "image_dimensions": 192,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/4": {
            "vector_length": 256,
            "image_dimensions": 160,
        },
        "https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/4": {
            "vector_length": 256,
            "image_dimensions": 128,
        },
        "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/4": {
            "vector_length": 1536
        },
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/4": {
            "vector_length": 1024,
        },
    }

    def _get_model(self, model: Any) -> ModelBase:
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
            if model in self.TFHUB_MODELS:
                from relevanceai.operations_new.vectorize.models.image.tfhub import (
                    TFHubImage2Vec,
                )

                vector_length = self.TFHUB_MODELS[model]["vector_length"]
                image_dimensions = (
                    self.TFHUB_MODELS[model]["image_dimensions"]
                    if "image_dimensions" in self.TFHUB_MODELS[model]
                    else None
                )
                model = TFHubImage2Vec(
                    url=model,
                    vector_length=vector_length,
                    image_dimensions=image_dimensions,
                )

                return model

            if model in self.CLIP_MODELS:
                from relevanceai.operations_new.vectorize.models.image.clip import (
                    ClipImage2Vec,
                )

                vector_length = self.CLIP_MODELS[model]["vector_length"]
                model = ClipImage2Vec(
                    url=model,
                    vector_length=vector_length,
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
