"""
A Collection of Image Tools For Relevance AI Utilies.

You can use as such:

.. code-block::

    from relevanceai.data_tools.base_image_processing import ImageTools
    # Now you want to READ it
    tools = ImageTools()
    arr = tools.read(image)

"""

try:
    import numpy as np
    import io
    import imageio
    import numpy as np
    import matplotlib.pyplot as plt
    from urllib.request import urlopen, Request
    from requests.exceptions import MissingSchema
    from skimage import transform
    from PIL import Image
    import requests
except ModuleNotFoundError:
    pass

try:
    from functools import singledispatchmethod  # type: ignore
except ImportError:
    from relevanceai.unstructured_data.utils.functools_extend import (
        singledispatchmethod,
    )


class ImageTools:
    def read(self, image):
        """
        An method to read images (converting them into NumPy arrays)
        Args:
            image: An image link/bytes/io Bytesio data format.
            as_gray: read in the image as black and white
        """
        im_arr = self._read(image)
        return self.process_image(im_arr)

    @singledispatchmethod
    def _read(self, image):
        pass

    @_read.register(str)
    def _(self, image: str):
        if "http" in image:
            try:
                b = io.BytesIO(
                    urlopen(
                        Request(image, headers={"User-Agent": "Mozilla/5.0"})
                    ).read()
                )
            except:
                import tensorflow as tf

                return tf.image.decode_jpeg(
                    requests.get(image).content, channels=3, name="jpeg_reader"
                ).numpy()
        else:
            # assume some default image processing is fine
            b = image  # type: ignore
        return b

    @_read.register(bytes)  # type: ignore
    def _(self, image):
        return io.BytesIO(image)

    @_read.register(io.BytesIO)  # type: ignore
    def _(self, image):
        return image

    @staticmethod
    def process_image(b):
        try:
            return np.array(imageio.imread(b, pilmode="RGB"))
        except:
            return np.array(imageio.imread(b)[:, :, :3])

    def is_greyscale(self, img_path: str):
        """Determine if an image is grayscale or not"""
        try:
            img = Image.open(requests.get(img_path, stream=True).raw)
        except MissingSchema:
            img = Image.open(img_path)
        img = img.convert("RGB")
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True

    def to_grayscale(self, sample, rgb_weights: list = None):
        """
        Converting an image from RGB to Grayscale
        """
        if rgb_weights is None:
            return np.repeat(
                np.dot(sample[..., :3], self.rgb_weights)[..., np.newaxis], 3, -1
            )
        else:
            return np.repeat(
                np.dot(sample[..., :3], rgb_weights)[..., np.newaxis], 3, -1
            )

    @property
    def rgb_weights(self):
        """
        Get RGB weights for grayscaling.
        """
        return [0.2989, 0.5870, 0.1140]

    def show_image(self, sample, cmap=None, is_grayscale=True):
        """
        Show an image once it is read.
        Arg:
            sample: Image that is read (numpy array)
        """
        if is_grayscale:
            return plt.imshow(sample, cmap=plt.get_cmap("gray"))
        return plt.imshow(sample, cmap=cmap)

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
