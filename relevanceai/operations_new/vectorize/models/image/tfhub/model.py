import io
import imageio
import requests

from typing import List, Any, Optional
from urllib.request import Request, urlopen
from skimage import transform

import numpy as np

from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.utils.decorators.vectors import catch_errors

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Run `pip install tensorflow_hub`.")
except:
    import traceback

    traceback.print_exc()


class TFHubImage2Vec(VectorizeModelBase):
    def __init__(self, url, vector_length, image_dimensions: Optional[int] = None):

        self.model: Any = hub.load(url)
        self.vector_length: int = vector_length
        self.image_dimensions: Optional[int] = image_dimensions
        self.model_name = self._get_model_name(url)

    def read(self, image: Any):
        """It takes an image as a string, bytes, or BytesIO object, and returns a numpy array of the image

        Parameters
        ----------
        image : str
            The image to be read.

        Returns
        -------
            The image is being returned as a numpy array.

        """
        if isinstance(image, str):
            if "http" in image:
                try:
                    b = io.BytesIO(
                        urlopen(
                            Request(image, headers={"User-Agent": "Mozilla/5.0"})
                        ).read()
                    )
                except:
                    return tf.image.decode_jpeg(
                        requests.get(image).content, channels=3, name="jpeg_reader"
                    ).numpy()
            else:
                b = image  # type: ignore
        elif isinstance(image, bytes):
            b = io.BytesIO(image)
        elif isinstance(image, io.BytesIO):
            b = image
        else:
            raise ValueError(
                "Cannot process data type. Ensure it is is string/bytes or BytesIO."
            )
        try:
            return np.array(imageio.imread(b, pilmode="RGB"))
        except:
            return np.array(imageio.imread(b)[:, :, :3])

    def image_resize(
        self, image_array, width=0, height=0, rescale=0, resize_mode="symmetric"
    ):
        if width and height:
            image_array = transform.resize(
                image_array, (width, height), mode=resize_mode, preserve_range=True
            )
        if rescale:
            image_array = transform.rescale(
                image_array, rescale, preserve_range=True, anti_aliasing=True
            )
        return np.array(image_array)

    @catch_errors
    def encode(self, image: str) -> List[float]:
        """It takes an image as input, resizes it to the dimensions specified in the model, and returns the
        output of the model.

        Parameters
        ----------
        image : str
            str

        Returns
        -------
            A list of floats.

        """
        if isinstance(image, str):
            image_tensor = self.read(image)

        if self.image_dimensions is not None:
            if image_tensor.shape[0] != self.image_dimensions:
                raise ValueError(
                    f"Incorrect Image size, please resize to {self.image_dimensions}x{self.image_dimensions}"
                )

        return self.model(image_tensor).numpy().tolist()[0]

    @catch_errors
    def bulk_encode(self, images: List[str]) -> List[List[float]]:
        """The function takes a list of strings and returns a list of lists of floats

        Parameters1
        ----------
        texts : List[str]
            List[str]

        Returns
        -------
            A list of lists of floats.

        """
        images = np.array([self.read(image) for image in images])
        return self.model(images).numpy().tolist()
